import { ClaudeDevProvider } from '../providers/ClaudeDevProvider';
import { HistoryItem } from '../shared/HistoryItem';
import { ClaudeMessage } from '../shared/ExtensionMessage';
import { Anthropic } from '@anthropic-ai/sdk';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as vscode from 'vscode';
import { findLastIndex } from '../utils';

export class TaskManager {
    private taskId?: string;
    private provider: ClaudeDevProvider;
    private historyItem?: HistoryItem;
    private task?: string;
    private images?: string[];
    private apiConversationHistory: Anthropic.MessageParam[] = [];
    private claudeMessages: ClaudeMessage[] = [];
    private abort: boolean = false;

    constructor(provider: ClaudeDevProvider, historyItem?: HistoryItem, task?: string, images?: string[]) {
        this.provider = provider;
        this.historyItem = historyItem;
        this.task = task;
        this.images = images;
        this.taskId = undefined;
    }

	getTaskId(): string {
        return this.taskId || '';
	}
    
    public initializeTask(): void {
        if (this.historyItem) {
            this.taskId = this.historyItem.id
            this.resumeTaskFromHistory();
        } else if (this.task || this.images) {
            this.startTask(this.task, this.images);
        } else {
            throw new Error("Either historyItem or task/images must be provided");
        }
    }

    public async startTask(task?: string, images?: string[]): Promise<void> {
        // conversationHistory (for API) and claudeMessages (for webview) need to be in sync
		// if the extension process were killed, then on restart the claudeMessages might not be empty, so we need to set it to [] when we create a new ClaudeDev client (otherwise webview would show stale messages from previous session)
		this.claudeMessages = []
		this.apiConversationHistory = []
		await this.providerRef.deref()?.postStateToWebview()

		await this.say("text", task, images)

		let imageBlocks: Anthropic.ImageBlockParam[] = this.formatImagesIntoBlocks(images)
		await this.initiateTaskLoop([
			{
				type: "text",
				text: `<task>\n${task}\n</task>`,
			},
			...imageBlocks,
		])
    }

    public async resumeTaskFromHistory(): Promise<void> {
        const modifiedClaudeMessages = await this.getSavedClaudeMessages()

		// Need to modify claude messages for good ux, i.e. if the last message is an api_request_started, then remove it otherwise the user will think the request is still loading
		const lastApiReqStartedIndex = modifiedClaudeMessages.reduce(
			(lastIndex, m, index) => (m.type === "say" && m.say === "api_req_started" ? index : lastIndex),
			-1
		)
		const lastApiReqFinishedIndex = modifiedClaudeMessages.reduce(
			(lastIndex, m, index) => (m.type === "say" && m.say === "api_req_finished" ? index : lastIndex),
			-1
		)
		if (lastApiReqStartedIndex > lastApiReqFinishedIndex && lastApiReqStartedIndex !== -1) {
			modifiedClaudeMessages.splice(lastApiReqStartedIndex, 1)
		}

		// Remove any resume messages that may have been added before
		const lastRelevantMessageIndex = findLastIndex(
			modifiedClaudeMessages,
			(m) => !(m.ask === "resume_task" || m.ask === "resume_completed_task")
		)
		if (lastRelevantMessageIndex !== -1) {
			modifiedClaudeMessages.splice(lastRelevantMessageIndex + 1)
		}

		await this.overwriteClaudeMessages(modifiedClaudeMessages)
		this.claudeMessages = await this.getSavedClaudeMessages()

		// Now present the claude messages to the user and ask if they want to resume

		const lastClaudeMessage = this.claudeMessages
			.slice()
			.reverse()
			.find((m) => !(m.ask === "resume_task" || m.ask === "resume_completed_task")) // could be multiple resume tasks
		// const lastClaudeMessage = this.claudeMessages[lastClaudeMessageIndex]
		// could be a completion result with a command
		// const secondLastClaudeMessage = this.claudeMessages
		// 	.slice()
		// 	.reverse()
		// 	.find(
		// 		(m, index) =>
		// 			index !== lastClaudeMessageIndex && !(m.ask === "resume_task" || m.ask === "resume_completed_task")
		// 	)
		// (lastClaudeMessage?.ask === "command" && secondLastClaudeMessage?.ask === "completion_result")

		let askType: ClaudeAsk
		if (lastClaudeMessage?.ask === "completion_result") {
			askType = "resume_completed_task"
		} else {
			askType = "resume_task"
		}

		const { response, text, images } = await this.ask(askType) // calls poststatetowebview
		let responseText: string | undefined
		let responseImages: string[] | undefined
		if (response === "messageResponse") {
			await this.say("user_feedback", text, images)
			responseText = text
			responseImages = images
		}

		// need to make sure that the api conversation history can be resumed by the api, even if it goes out of sync with claude messages

		// if the last message is an assistant message, we need to check if there's tool use since every tool use has to have a tool response
		// if there's no tool use and only a text block, then we can just add a user message

		// if the last message is a user message, we can need to get the assistant message before it to see if it made tool calls, and if so, fill in the remaining tool responses with 'interrupted'

		const existingApiConversationHistory: Anthropic.Messages.MessageParam[] =
			await this.getSavedApiConversationHistory()

		let modifiedOldUserContent: UserContent // either the last message if its user message, or the user message before the last (assistant) message
		let modifiedApiConversationHistory: Anthropic.Messages.MessageParam[] // need to remove the last user message to replace with new modified user message
		if (existingApiConversationHistory.length > 0) {
			const lastMessage = existingApiConversationHistory[existingApiConversationHistory.length - 1]

			if (lastMessage.role === "assistant") {
				const content = Array.isArray(lastMessage.content)
					? lastMessage.content
					: [{ type: "text", text: lastMessage.content }]
				const hasToolUse = content.some((block) => block.type === "tool_use")

				if (hasToolUse) {
					const toolUseBlocks = content.filter(
						(block) => block.type === "tool_use"
					) as Anthropic.Messages.ToolUseBlock[]
					const toolResponses: Anthropic.ToolResultBlockParam[] = toolUseBlocks.map((block) => ({
						type: "tool_result",
						tool_use_id: block.id,
						content: "Task was interrupted before this tool call could be completed.",
					}))
					modifiedApiConversationHistory = [...existingApiConversationHistory] // no changes
					modifiedOldUserContent = [...toolResponses]
				} else {
					modifiedApiConversationHistory = [...existingApiConversationHistory]
					modifiedOldUserContent = []
				}
			} else if (lastMessage.role === "user") {
				const previousAssistantMessage: Anthropic.Messages.MessageParam | undefined =
					existingApiConversationHistory[existingApiConversationHistory.length - 2]

				const existingUserContent: UserContent = Array.isArray(lastMessage.content)
					? lastMessage.content
					: [{ type: "text", text: lastMessage.content }]
				if (previousAssistantMessage && previousAssistantMessage.role === "assistant") {
					const assistantContent = Array.isArray(previousAssistantMessage.content)
						? previousAssistantMessage.content
						: [{ type: "text", text: previousAssistantMessage.content }]

					const toolUseBlocks = assistantContent.filter(
						(block) => block.type === "tool_use"
					) as Anthropic.Messages.ToolUseBlock[]

					if (toolUseBlocks.length > 0) {
						const existingToolResults = existingUserContent.filter(
							(block) => block.type === "tool_result"
						) as Anthropic.ToolResultBlockParam[]

						const missingToolResponses: Anthropic.ToolResultBlockParam[] = toolUseBlocks
							.filter(
								(toolUse) => !existingToolResults.some((result) => result.tool_use_id === toolUse.id)
							)
							.map((toolUse) => ({
								type: "tool_result",
								tool_use_id: toolUse.id,
								content: "Task was interrupted before this tool call could be completed.",
							}))

						modifiedApiConversationHistory = existingApiConversationHistory.slice(0, -1) // removes the last user message
						modifiedOldUserContent = [...existingUserContent, ...missingToolResponses]
					} else {
						modifiedApiConversationHistory = existingApiConversationHistory.slice(0, -1)
						modifiedOldUserContent = [...existingUserContent]
					}
				} else {
					modifiedApiConversationHistory = existingApiConversationHistory.slice(0, -1)
					modifiedOldUserContent = [...existingUserContent]
				}
			} else {
				throw new Error("Unexpected: Last message is not a user or assistant message")
			}
		} else {
			throw new Error("Unexpected: No existing API conversation history")
		}

		let newUserContent: UserContent = [...modifiedOldUserContent]

		const agoText = (() => {
			const timestamp = lastClaudeMessage?.ts ?? Date.now()
			const now = Date.now()
			const diff = now - timestamp
			const minutes = Math.floor(diff / 60000)
			const hours = Math.floor(minutes / 60)
			const days = Math.floor(hours / 24)

			if (days > 0) {
				return `${days} day${days > 1 ? "s" : ""} ago`
			}
			if (hours > 0) {
				return `${hours} hour${hours > 1 ? "s" : ""} ago`
			}
			if (minutes > 0) {
				return `${minutes} minute${minutes > 1 ? "s" : ""} ago`
			}
			return "just now"
		})()

		newUserContent.push({
			type: "text",
			text:
				`Task resumption: This autonomous coding task was interrupted ${agoText}. It may or may not be complete, so please reassess the task context. Be aware that the project state may have changed since then. The current working directory is now '${cwd.toPosix()}'. If the task has not been completed, retry the last step before interruption and proceed with completing the task.` +
				(responseText
					? `\n\nNew instructions for task continuation:\n<user_message>\n${responseText}\n</user_message>`
					: ""),
		})

		if (responseImages && responseImages.length > 0) {
			newUserContent.push(...this.formatImagesIntoBlocks(responseImages))
		}

		await this.overwriteApiConversationHistory(modifiedApiConversationHistory)
		await this.initiateTaskLoop(newUserContent)
    }

    public abortTask(): void {
        this.abort = true;
        // Additional cleanup if needed
    }

    public async ensureTaskDirectoryExists(): Promise<string> {
        const globalStoragePath = this.provider.context.globalStorageUri.fsPath;
        const taskDir = path.join(globalStoragePath, "tasks", this.historyItem?.id || Date.now().toString());
        await fs.mkdir(taskDir, { recursive: true });
        return taskDir;
    }

    public async getSavedApiConversationHistory(): Promise<Anthropic.MessageParam[]> {
        const filePath = path.join(await this.ensureTaskDirectoryExists(), "api_conversation_history.json");
        const fileExists = await fs.access(filePath).then(() => true).catch(() => false);
        if (fileExists) {
            return JSON.parse(await fs.readFile(filePath, "utf8"));
        }
        return [];
    }

    public async addToApiConversationHistory(message: Anthropic.MessageParam): Promise<void> {
        this.apiConversationHistory.push(message);
        await this.saveApiConversationHistory();
    }

    public async overwriteApiConversationHistory(newHistory: Anthropic.MessageParam[]): Promise<void> {
        this.apiConversationHistory = newHistory;
        await this.saveApiConversationHistory();
    }

    public async saveApiConversationHistory(): Promise<void> {
        try {
            const filePath = path.join(await this.ensureTaskDirectoryExists(), "api_conversation_history.json");
            await fs.writeFile(filePath, JSON.stringify(this.apiConversationHistory));
        } catch (error) {
            console.error("Failed to save API conversation history:", error);
        }
    }

    public async getSavedClaudeMessages(): Promise<ClaudeMessage[]> {
        const filePath = path.join(await this.ensureTaskDirectoryExists(), "claude_messages.json");
        const fileExists = await fs.access(filePath).then(() => true).catch(() => false);
        if (fileExists) {
            return JSON.parse(await fs.readFile(filePath, "utf8"));
        }
        return [];
    }

    public async addToClaudeMessages(message: ClaudeMessage): Promise<void> {
        this.claudeMessages.push(message);
        await this.saveClaudeMessages();
    }

    public async overwriteClaudeMessages(newMessages: ClaudeMessage[]): Promise<void> {
        this.claudeMessages = newMessages;
        await this.saveClaudeMessages();
    }

    public async saveClaudeMessages(): Promise<void> {
        try {
            const filePath = path.join(await this.ensureTaskDirectoryExists(), "claude_messages.json");
            await fs.writeFile(filePath, JSON.stringify(this.claudeMessages));
        } catch (error) {
            console.error("Failed to save claude messages:", error);
        }
    }

    // Additional methods will be implemented here
}
