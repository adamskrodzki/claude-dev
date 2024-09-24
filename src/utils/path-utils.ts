
import * as diff from "diff"
import * as vscode from "vscode"
import { arePathsEqual } from "./path-helpers"
import path from "path"
import os from "os"
import { Anthropic } from "@anthropic-ai/sdk"


export type ToolResponse = string | Array<Anthropic.TextBlockParam | Anthropic.ImageBlockParam>

export const cwd =
	vscode.workspace.workspaceFolders?.map((folder) => folder.uri.fsPath).at(0) ?? path.join(os.homedir(), "Desktop") // may or may not exist but fs checking existence would immediately ask for permission which would be bad UX, need to come up with a better solution


export function createPrettyPatch(filename = "file", oldStr: string, newStr: string) {
    const patch = diff.createPatch(filename.toPosix(), oldStr, newStr)
    const lines = patch.split("\n")
    const prettyPatchLines = lines.slice(4)
    return prettyPatchLines.join("\n")
}

export function getReadablePath(relPath: string): string {
    // path.resolve is flexible in that it will resolve relative paths like '../../' to the cwd and even ignore the cwd if the relPath is actually an absolute path
    const absolutePath = path.resolve(cwd, relPath)
    if (arePathsEqual(cwd, path.join(os.homedir(), "Desktop"))) {
        // User opened vscode without a workspace, so cwd is the Desktop. Show the full absolute path to keep the user aware of where files are being created
        return absolutePath.toPosix()
    }
    if (arePathsEqual(path.normalize(absolutePath), path.normalize(cwd))) {
        return path.basename(absolutePath).toPosix()
    } else {
        // show the relative path to the cwd
        const normalizedRelPath = path.relative(cwd, absolutePath)
        if (absolutePath.includes(cwd)) {
            return normalizedRelPath.toPosix()
        } else {
            // we are outside the cwd, so show the absolute path (useful for when claude passes in '../../' for example)
            return absolutePath.toPosix()
        }
    }
}
