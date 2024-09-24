import { createPrettyPatch, getReadablePath, ToolResponse, cwd, formatFilesList } from "./utils"
import { diagnosticsToProblemsString, getNewDiagnostics } from "./utils/diagnostics"
import * as vscode from "vscode"
import fs from "fs/promises"
import * as diff from "diff"
import * as path from "path"
import { arePathsEqual } from "./utils/path-helpers"
import { ClaudeAskResponse } from "./shared/WebviewMessage"
import { ClaudeAsk, ClaudeSay, ClaudeSayTool } from "./shared/ExtensionMessage"
import { serializeError } from "serialize-error"
import { EnvironmentManager } from "./EnvironmentManager"
import { ToolName } from "./shared/Tool"
import { formatToolResponseWithImages } from "./utils/openai-format"

import { listFiles } from "./parse-source-code"
import { extractTextFromFile } from "./utils/extract-text"

export class FileSystemHandler{
    consecutiveMistakeCount: number
    environmentManager: EnvironmentManager
    images? : string[]
    say : (type: ClaudeSay, text?: string, images?: string[]) => Promise<undefined>
    ask : (type: ClaudeAsk,question?: string) => Promise<{ response: ClaudeAskResponse; text?: string; images?: string[] }>
	allowReadOnly: boolean
    
    constructor(environmentManager: EnvironmentManager, 
        say : (type: ClaudeSay, text?: string, images?: string[]) => Promise<undefined>,
        ask : (type: ClaudeAsk, question?: string) =>  Promise<{ response: ClaudeAskResponse; text?: string; images?: string[] }>,
		allowReadOnly : boolean
	) {
        this.consecutiveMistakeCount = 0;
        this.environmentManager = environmentManager;
        this.say = say;
        this.ask = ask;
		this.allowReadOnly = allowReadOnly;
    }
    
	// return is [didUserRejectTool, ToolResponse]
	async writeToFile(relPath?: string, newContent?: string): Promise<[boolean, ToolResponse]> {
		if (relPath === undefined) {
			this.consecutiveMistakeCount++
			return [false, await this.sayAndCreateMissingParamError("write_to_file", "path")]
		}
		if (newContent === undefined) {
			this.consecutiveMistakeCount++
			// Custom error message for this particular case
			await this.say(
				"error",
				`Claude tried to use write_to_file for '${relPath.toPosix()}' without value for required parameter 'content'. This is likely due to reaching the maximum output token limit. Retrying with suggestion to change response size...`
			)
			return [
				false,
				await this.formatToolError(
					`Missing value for required parameter 'content'. This may occur if the file is too large, exceeding output limits. Consider splitting into smaller files or reducing content size. Please retry with all required parameters.`
				),
			]
		}
		this.consecutiveMistakeCount = 0
		try {
			const absolutePath = path.resolve(cwd, relPath)
			const fileExists = await fs
				.access(absolutePath)
				.then(() => true)
				.catch(() => false)

			// if the file is already open, ensure it's not dirty before getting its contents
			if (fileExists) {
				const existingDocument = vscode.workspace.textDocuments.find((doc) =>
					arePathsEqual(doc.uri.fsPath, absolutePath)
				)
				if (existingDocument && existingDocument.isDirty) {
					await existingDocument.save()
				}
			}

			// get diagnostics before editing the file, we'll compare to diagnostics after editing to see if claude needs to fix anything
			const preDiagnostics = vscode.languages.getDiagnostics()

			let originalContent: string
			if (fileExists) {
				originalContent = await fs.readFile(absolutePath, "utf-8")
				// fix issue where claude always removes newline from the file
				const eol = originalContent.includes("\r\n") ? "\r\n" : "\n"
				if (originalContent.endsWith(eol) && !newContent.endsWith(eol)) {
					newContent += eol
				}
			} else {
				originalContent = ""
			}

			const fileName = path.basename(absolutePath)

			// for new files, create any necessary directories and keep track of new directories to delete if the user denies the operation

			// Keep track of newly created directories
			const createdDirs: string[] = await this.createDirectoriesForFile(absolutePath)
			// console.log(`Created directories: ${createdDirs.join(", ")}`)
			// make sure the file exists before we open it
			if (!fileExists) {
				await fs.writeFile(absolutePath, "")
			}

			// Open the existing file with the new contents
			const updatedDocument = await vscode.workspace.openTextDocument(vscode.Uri.file(absolutePath))

			// await updatedDocument.save()
			// const edit = new vscode.WorkspaceEdit()
			// const fullRange = new vscode.Range(
			// 	updatedDocument.positionAt(0),
			// 	updatedDocument.positionAt(updatedDocument.getText().length)
			// )
			// edit.replace(updatedDocument.uri, fullRange, newContent)
			// await vscode.workspace.applyEdit(edit)

			// Windows file locking issues can prevent temporary files from being saved or closed properly.
			// To avoid these problems, we use in-memory TextDocument objects with the `untitled` scheme.
			// This method keeps the document entirely in memory, bypassing the filesystem and ensuring
			// a consistent editing experience across all platforms. This also has the added benefit of not
			// polluting the user's workspace with temporary files.

			// Create an in-memory document for the new content
			// const inMemoryDocumentUri = vscode.Uri.parse(`untitled:${fileName}`) // untitled scheme is necessary to open a file without it being saved to disk
			// const inMemoryDocument = await vscode.workspace.openTextDocument(inMemoryDocumentUri)
			// const edit = new vscode.WorkspaceEdit()
			// edit.insert(inMemoryDocumentUri, new vscode.Position(0, 0), newContent)
			// await vscode.workspace.applyEdit(edit)

			// Show diff
			await vscode.commands.executeCommand(
				"vscode.diff",
				vscode.Uri.parse(`claude-dev-diff:${fileName}`).with({
					query: Buffer.from(originalContent).toString("base64"),
				}),
				updatedDocument.uri,
				`${fileName}: ${fileExists ? "Original ↔ Claude's Changes" : "New File"} (Editable)`
			)

			// if the file was already open, close it (must happen after showing the diff view since if it's the only tab the column will close)
			let documentWasOpen = false

			// close the tab if it's open
			const tabs = vscode.window.tabGroups.all
				.map((tg) => tg.tabs)
				.flat()
				.filter(
					(tab) =>
						tab.input instanceof vscode.TabInputText && arePathsEqual(tab.input.uri.fsPath, absolutePath)
				)
			for (const tab of tabs) {
				await vscode.window.tabGroups.close(tab)
				// console.log(`Closed tab for ${absolutePath}`)
				documentWasOpen = true
			}

			// console.log(`Document was open: ${documentWasOpen}`)

			// edit needs to happen after we close the original tab
			const edit = new vscode.WorkspaceEdit()
			if (!fileExists) {
				edit.insert(updatedDocument.uri, new vscode.Position(0, 0), newContent)
			} else {
				const fullRange = new vscode.Range(
					updatedDocument.positionAt(0),
					updatedDocument.positionAt(updatedDocument.getText().length)
				)
				edit.replace(updatedDocument.uri, fullRange, newContent)
			}
			// Apply the edit, but without saving so this doesnt trigger a local save in timeline history
			await vscode.workspace.applyEdit(edit) // has the added benefit of maintaing the file's original EOLs

			// Find the first range where the content differs and scroll to it
			if (fileExists) {
				const diffResult = diff.diffLines(originalContent, newContent)
				for (let i = 0, lineCount = 0; i < diffResult.length; i++) {
					const part = diffResult[i]
					if (part.added || part.removed) {
						const startLine = lineCount + 1
						const endLine = lineCount + (part.count || 0)
						const activeEditor = vscode.window.activeTextEditor
						if (activeEditor) {
							try {
								activeEditor.revealRange(
									// + 3 to move the editor up slightly as this looks better
									new vscode.Range(
										new vscode.Position(startLine, 0),
										new vscode.Position(
											Math.min(endLine + 3, activeEditor.document.lineCount - 1),
											0
										)
									),
									vscode.TextEditorRevealType.InCenter
								)
							} catch (error) {
								console.error(`Error revealing range for ${absolutePath}: ${error}`)
							}
						}
						break
					}
					lineCount += part.count || 0
				}
			}

			// remove cursor from the document
			await vscode.commands.executeCommand("workbench.action.focusSideBar")

			let userResponse: {
				response: ClaudeAskResponse
				text?: string
				images?: string[]
			}
			if (fileExists) {
				userResponse = await this.ask(
					"tool",
					JSON.stringify({
						tool: "editedExistingFile",
						path: getReadablePath(relPath),
						diff: createPrettyPatch(relPath, originalContent, newContent),
					} satisfies ClaudeSayTool)
				)
			} else {
				userResponse = await this.ask(
					"tool",
					JSON.stringify({
						tool: "newFileCreated",
						path: getReadablePath(relPath),
						content: newContent,
					} satisfies ClaudeSayTool)
				)
			}
			const { response, text, images } = userResponse

			if (response !== "yesButtonTapped") {
				if (!fileExists) {
					if (updatedDocument.isDirty) {
						await updatedDocument.save()
					}
					await this.closeDiffViews()
					await fs.unlink(absolutePath)
					// Remove only the directories we created, in reverse order
					for (let i = createdDirs.length - 1; i >= 0; i--) {
						await fs.rmdir(createdDirs[i])
						console.log(`Directory ${createdDirs[i]} has been deleted.`)
					}
					console.log(`File ${absolutePath} has been deleted.`)
				} else {
					// revert document
					const edit = new vscode.WorkspaceEdit()
					const fullRange = new vscode.Range(
						updatedDocument.positionAt(0),
						updatedDocument.positionAt(updatedDocument.getText().length)
					)
					edit.replace(updatedDocument.uri, fullRange, originalContent)
					// Apply the edit and save, since contents shouldnt have changed this wont show in local history unless of course the user made changes and saved during the edit
					await vscode.workspace.applyEdit(edit)
					await updatedDocument.save()
					console.log(`File ${absolutePath} has been reverted to its original content.`)
					if (documentWasOpen) {
						await vscode.window.showTextDocument(vscode.Uri.file(absolutePath), { preview: false })
					}
					await this.closeDiffViews()
				}

				if (response === "messageResponse") {
					await this.say("user_feedback", text, images)
					return [true, formatToolResponseWithImages(await this.formatToolDeniedFeedback(text), images)]
				}
				return [true, await this.formatToolDenied()]
			}

			// Save the changes
			const editedContent = updatedDocument.getText()
			if (updatedDocument.isDirty) {
				await updatedDocument.save()
			}
			this.environmentManager.setFileEdit(true);

			// Read the potentially edited content from the document

			// trigger an entry in the local history for the file
			// if (fileExists) {
			// 	await fs.writeFile(absolutePath, originalContent)
			// 	const editor = await vscode.window.showTextDocument(vscode.Uri.file(absolutePath), { preview: false })
			// 	const edit = new vscode.WorkspaceEdit()
			// 	const fullRange = new vscode.Range(
			// 		editor.document.positionAt(0),
			// 		editor.document.positionAt(editor.document.getText().length)
			// 	)
			// 	edit.replace(editor.document.uri, fullRange, editedContent)
			// 	// Apply the edit, this will trigger a local save and timeline history
			// 	await vscode.workspace.applyEdit(edit) // has the added benefit of maintaing the file's original EOLs
			// 	await editor.document.save()
			// }

			// if (!fileExists) {
			// 	await fs.mkdir(path.dirname(absolutePath), { recursive: true })
			// 	await fs.writeFile(absolutePath, "")
			// }
			// await closeInMemoryDocAndDiffViews()

			// await fs.writeFile(absolutePath, editedContent)

			// open file and add text to it, if it fails fallback to using writeFile
			// we try doing it this way since it adds to local history for users to see what's changed in the file's timeline
			// try {
			// 	const editor = await vscode.window.showTextDocument(vscode.Uri.file(absolutePath), { preview: false })
			// 	const edit = new vscode.WorkspaceEdit()
			// 	const fullRange = new vscode.Range(
			// 		editor.document.positionAt(0),
			// 		editor.document.positionAt(editor.document.getText().length)
			// 	)
			// 	edit.replace(editor.document.uri, fullRange, editedContent)
			// 	// Apply the edit, this will trigger a local save and timeline history
			// 	await vscode.workspace.applyEdit(edit) // has the added benefit of maintaing the file's original EOLs
			// 	await editor.document.save()
			// } catch (saveError) {
			// 	console.log(`Could not open editor for ${absolutePath}: ${saveError}`)
			// 	await fs.writeFile(absolutePath, editedContent)
			// 	// calling showTextDocument would sometimes fail even though changes were applied, so we'll ignore these one-off errors (likely due to vscode locking issues)
			// 	try {
			// 		await vscode.window.showTextDocument(vscode.Uri.file(absolutePath), { preview: false })
			// 	} catch (openFileError) {
			// 		console.log(`Could not open editor for ${absolutePath}: ${openFileError}`)
			// 	}
			// }

			await vscode.window.showTextDocument(vscode.Uri.file(absolutePath), { preview: false })

			await this.closeDiffViews()

			/*
			Getting diagnostics before and after the file edit is a better approach than
			automatically tracking problems in real-time. This method ensures we only
			report new problems that are a direct result of this specific edit.
			Since these are new problems resulting from Claude's edit, we know they're
			directly related to the work he's doing. This eliminates the risk of Claude
			going off-task or getting distracted by unrelated issues, which was a problem
			with the previous auto-debug approach. Some users' machines may be slow to
			update diagnostics, so this approach provides a good balance between automation
			and avoiding potential issues where Claude might get stuck in loops due to
			outdated problem information. If no new problems show up by the time the user
			accepts the changes, they can always debug later using the '@problems' mention.
			This way, Claude only becomes aware of new problems resulting from his edits
			and can address them accordingly. If problems don't change immediately after
			applying a fix, Claude won't be notified, which is generally fine since the
			initial fix is usually correct and it may just take time for linters to catch up.
			*/
			const postDiagnostics = vscode.languages.getDiagnostics()
			const newProblems = diagnosticsToProblemsString(getNewDiagnostics(preDiagnostics, postDiagnostics), cwd) // will be empty string if no errors/warnings
			const newProblemsMessage =
				newProblems.length > 0 ? `\n\nNew problems detected after saving the file:\n${newProblems}` : ""
			// await vscode.window.showTextDocument(vscode.Uri.file(absolutePath), { preview: false })

			// If the edited content has different EOL characters, we don't want to show a diff with all the EOL differences.
			const newContentEOL = newContent.includes("\r\n") ? "\r\n" : "\n"
			const normalizedEditedContent = editedContent.replace(/\r\n|\n/g, newContentEOL)
			const normalizedNewContent = newContent.replace(/\r\n|\n/g, newContentEOL) // just in case the new content has a mix of varying EOL characters
			if (normalizedEditedContent !== normalizedNewContent) {
				const userDiff = diff.createPatch(relPath.toPosix(), normalizedNewContent, normalizedEditedContent)
				await this.say(
					"user_feedback_diff",
					JSON.stringify({
						tool: fileExists ? "editedExistingFile" : "newFileCreated",
						path: getReadablePath(relPath),
						diff: createPrettyPatch(relPath, normalizedNewContent, normalizedEditedContent),
					} satisfies ClaudeSayTool)
				)
				return [
					false,
					await this.formatToolResult(
						`The user made the following updates to your content:\n\n${userDiff}\n\nThe updated content, which includes both your original modifications and the user's additional edits, has been successfully saved to ${relPath.toPosix()}. (Note this does not mean you need to re-write the file with the user's changes, as they have already been applied to the file.)${newProblemsMessage}`
					),
				]
			} else {
				return [
					false,
					await this.formatToolResult(
						`The content was successfully saved to ${relPath.toPosix()}.${newProblemsMessage}`
					),
				]
			}
		} catch (error) {
			const errorString = `Error writing file: ${JSON.stringify(serializeError(error))}`
			await this.say(
				"error",
				`Error writing file:\n${error.message ?? JSON.stringify(serializeError(error), null, 2)}`
			)
			return [false, await this.formatToolError(errorString)]
		}
	}
	
	async readFile(relPath?: string): Promise<[boolean, ToolResponse]> {
		if (relPath === undefined) {
			this.consecutiveMistakeCount++
			return [false, await this.sayAndCreateMissingParamError("read_file", "path")]
		}
		this.consecutiveMistakeCount = 0
		try {
			const absolutePath = path.resolve(cwd, relPath)
			const content = await extractTextFromFile(absolutePath)

			const message = JSON.stringify({
				tool: "readFile",
				path: getReadablePath(relPath),
				content: absolutePath,
			} satisfies ClaudeSayTool)
			if (this.allowReadOnly) {
				await this.say("tool", message)
			} else {
				const { response, text, images } = await this.ask("tool", message)
				if (response !== "yesButtonTapped") {
					if (response === "messageResponse") {
						await this.say("user_feedback", text, images)
						return [
							true,
							formatToolResponseWithImages(await this.formatToolDeniedFeedback(text), images),
						]
					}
					return [true, await this.formatToolDenied()]
				}
			}

			return [false, content]
		} catch (error) {
			const errorString = `Error reading file: ${JSON.stringify(serializeError(error))}`
			await this.say(
				"error",
				`Error reading file:\n${error.message ?? JSON.stringify(serializeError(error), null, 2)}`
			)
			return [false, await this.formatToolError(errorString)]
		}
	}

	async listFiles(relDirPath?: string, recursiveRaw?: string): Promise<[boolean, ToolResponse]> {
		if (relDirPath === undefined) {
			this.consecutiveMistakeCount++
			return [false, await this.sayAndCreateMissingParamError("list_files", "path")]
		}
		this.consecutiveMistakeCount = 0
		try {
			const recursive = recursiveRaw?.toLowerCase() === "true"
			const absolutePath = path.resolve(cwd, relDirPath)
			const [files, didHitLimit] = await listFiles(absolutePath, recursive, 200)
			const result = formatFilesList(absolutePath, files, didHitLimit)

			const message = JSON.stringify({
				tool: recursive ? "listFilesRecursive" : "listFilesTopLevel",
				path: getReadablePath(relDirPath),
				content: result,
			} satisfies ClaudeSayTool)
			if (this.allowReadOnly) {
				await this.say("tool", message)
			} else {
				const { response, text, images } = await this.ask("tool", message)
				if (response !== "yesButtonTapped") {
					if (response === "messageResponse") {
						await this.say("user_feedback", text, images)
						return [
							true,
							formatToolResponseWithImages(await this.formatToolDeniedFeedback(text), images),
						]
					}
					return [true, await this.formatToolDenied()]
				}
			}

			return [false, await this.formatToolResult(result)]
		} catch (error) {
			const errorString = `Error listing files and directories: ${JSON.stringify(serializeError(error))}`
			await this.say(
				"error",
				`Error listing files and directories:\n${
					error.message ?? JSON.stringify(serializeError(error), null, 2)
				}`
			)
			return [false, await this.formatToolError(errorString)]
		}
	}
    
	async sayAndCreateMissingParamError(toolName: ToolName, paramName: string, relPath?: string) {
		await this.say(
			"error",
			`Claude tried to use ${toolName}${
				relPath ? ` for '${relPath.toPosix()}'` : ""
			} without value for required parameter '${paramName}'. Retrying...`
		)
		return await this.formatToolError(
			`Missing value for required parameter '${paramName}'. Please retry with complete response.`
		)
	}

	async formatToolError(error?: string) {
		return `The tool execution failed with the following error:\n<error>\n${error}\n</error>`
	}
	/**
	 * Asynchronously creates all non-existing subdirectories for a given file path
	 * and collects them in an array for later deletion.
	 *
	 * @param filePath - The full path to a file.
	 * @returns A promise that resolves to an array of newly created directories.
	 */
	async createDirectoriesForFile(filePath: string): Promise<string[]> {
		const newDirectories: string[] = []
		const normalizedFilePath = path.normalize(filePath) // Normalize path for cross-platform compatibility
		const directoryPath = path.dirname(normalizedFilePath)

		let currentPath = directoryPath
		const dirsToCreate: string[] = []

		// Traverse up the directory tree and collect missing directories
		while (!(await this.exists(currentPath))) {
			dirsToCreate.push(currentPath)
			currentPath = path.dirname(currentPath)
		}

		// Create directories from the topmost missing one down to the target directory
		for (let i = dirsToCreate.length - 1; i >= 0; i--) {
			await fs.mkdir(dirsToCreate[i])
			newDirectories.push(dirsToCreate[i])
		}

		return newDirectories
	}
    
	/**
	 * Helper function to check if a path exists.
	 *
	 * @param path - The path to check.
	 * @returns A promise that resolves to true if the path exists, false otherwise.
	 */
	async exists(filePath: string): Promise<boolean> {
		try {
			await fs.access(filePath)
			return true
		} catch {
			return false
		}
	}

    async closeDiffViews() {
		const tabs = vscode.window.tabGroups.all
			.map((tg) => tg.tabs)
			.flat()
			.filter(
				(tab) =>
					tab.input instanceof vscode.TabInputTextDiff && tab.input?.original?.scheme === "claude-dev-diff"
			)

		for (const tab of tabs) {
			// trying to close dirty views results in save popup
			if (!tab.isDirty) {
				await vscode.window.tabGroups.close(tab)
			}
		}
	}

	async formatToolDeniedFeedback(feedback?: string) {
		return `The user denied this operation and provided the following feedback:\n<feedback>\n${feedback}\n</feedback>`
	}

	async formatToolDenied() {
		return `The user denied this operation.`
	}

	async formatToolResult(result: string) {
		return result // the successful result of the tool should never be manipulated, if we need to add details it should be as a separate user text block
	}

}