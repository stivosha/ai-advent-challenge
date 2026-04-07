import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { execFile } from "child_process";
import { promisify } from "util";
import { resolve } from "path";

const execFileAsync = promisify(execFile);

// ─── Git runner ─────────────────────────────────────────────────────────────

/**
 * Run a git command in the given working directory.
 * Returns { stdout, stderr } or throws on non-zero exit.
 */
async function git(args, cwd) {
  const dir = resolve(cwd ?? process.cwd());
  try {
    const { stdout, stderr } = await execFileAsync("git", args, {
      cwd: dir,
      windowsHide: true,
      maxBuffer: 10 * 1024 * 1024, // 10 MB
    });
    return { stdout: stdout.trimEnd(), stderr: stderr.trimEnd() };
  } catch (err) {
    // execFile rejects with an Error that has .stdout / .stderr
    const msg = [
      err.stderr?.trim(),
      err.stdout?.trim(),
      err.message,
    ]
      .filter(Boolean)
      .join("\n");
    throw new Error(msg || String(err));
  }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function ok(text) {
  return { content: [{ type: "text", text: String(text || "(empty)") }] };
}

function joinOut({ stdout, stderr }) {
  const parts = [stdout, stderr].filter(Boolean);
  return parts.length ? parts.join("\n") : "(no output)";
}

// ─── Tool definitions ────────────────────────────────────────────────────────

const TOOLS = [
  // ── Repository lifecycle ─────────────────────────────────────────────────
  {
    name: "git_init",
    description: "Initialise a new Git repository. Pass `bare: true` for a bare repo.",
    inputSchema: {
      type: "object",
      properties: {
        path:        { type: "string",  description: "Directory to initialise (default: current directory)" },
        bare:        { type: "boolean", description: "Create a bare repository" },
        initial_branch: { type: "string", description: "Name of the initial branch (e.g. 'main')" },
      },
    },
    async handler({ path: p, bare, initial_branch }) {
      const args = ["init"];
      if (bare) args.push("--bare");
      if (initial_branch) args.push("-b", initial_branch);
      if (p) args.push(p);
      const res = await git(args, p ?? ".");
      return ok(joinOut(res));
    },
  },
  {
    name: "git_clone",
    description: "Clone a remote repository into a local directory.",
    inputSchema: {
      type: "object",
      required: ["url"],
      properties: {
        url:    { type: "string", description: "Repository URL or local path to clone from" },
        dir:    { type: "string", description: "Destination directory (default: repo name)" },
        depth:  { type: "number", description: "Create a shallow clone with that many commits" },
        branch: { type: "string", description: "Clone a specific branch" },
        cwd:    { type: "string", description: "Working directory in which to run the clone" },
      },
    },
    async handler({ url, dir, depth, branch, cwd }) {
      const args = ["clone"];
      if (depth) args.push("--depth", String(depth));
      if (branch) args.push("-b", branch);
      args.push(url);
      if (dir) args.push(dir);
      const res = await git(args, cwd ?? ".");
      return ok(joinOut(res));
    },
  },

  // ── Working-tree state ───────────────────────────────────────────────────
  {
    name: "git_status",
    description: "Show the working-tree status (staged, unstaged, untracked files).",
    inputSchema: {
      type: "object",
      properties: {
        cwd:   { type: "string",  description: "Repository path" },
        short: { type: "boolean", description: "Use short format" },
      },
    },
    async handler({ cwd, short }) {
      const args = ["status"];
      if (short) args.push("-s");
      const res = await git(args, cwd);
      return ok(joinOut(res));
    },
  },
  {
    name: "git_diff",
    description: "Show changes between commits, commit and working tree, or staged changes.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:    { type: "string",  description: "Repository path" },
        staged: { type: "boolean", description: "Show staged (--cached) diff" },
        commit: { type: "string",  description: "Commit or ref to diff against (e.g. HEAD~1)" },
        paths:  { type: "array",   items: { type: "string" }, description: "Limit diff to these paths" },
        stat:   { type: "boolean", description: "Show diffstat only, no patch content" },
      },
    },
    async handler({ cwd, staged, commit, paths, stat }) {
      const args = ["diff"];
      if (staged) args.push("--cached");
      if (stat)   args.push("--stat");
      if (commit) args.push(commit);
      if (paths?.length) args.push("--", ...paths);
      const res = await git(args, cwd);
      return ok(joinOut(res));
    },
  },

  // ── Staging ──────────────────────────────────────────────────────────────
  {
    name: "git_add",
    description: "Stage files for the next commit.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:   { type: "string", description: "Repository path" },
        paths: {
          type: "array",
          items: { type: "string" },
          description: "Files or patterns to stage. Omit (or use ['.']) for all changes.",
        },
        patch: { type: "boolean", description: "Interactively stage hunks (--patch). NOTE: non-interactive; will fail if git asks questions." },
      },
    },
    async handler({ cwd, paths, patch }) {
      const args = ["add"];
      if (patch) args.push("-p");
      args.push(...(paths?.length ? paths : ["."]));
      const res = await git(args, cwd);
      return ok(joinOut(res));
    },
  },
  {
    name: "git_restore",
    description: "Discard working-tree changes or unstage files.",
    inputSchema: {
      type: "object",
      required: ["paths"],
      properties: {
        cwd:    { type: "string", description: "Repository path" },
        paths:  { type: "array",  items: { type: "string" }, description: "Paths to restore" },
        staged: { type: "boolean", description: "Unstage (--staged)" },
        source: { type: "string",  description: "Restore from this tree-ish (e.g. HEAD)" },
      },
    },
    async handler({ cwd, paths, staged, source }) {
      const args = ["restore"];
      if (staged) args.push("--staged");
      if (source) args.push("--source", source);
      args.push("--", ...paths);
      const res = await git(args, cwd);
      return ok(joinOut(res));
    },
  },

  // ── Committing ───────────────────────────────────────────────────────────
  {
    name: "git_commit",
    description: "Record staged changes as a new commit.",
    inputSchema: {
      type: "object",
      required: ["message"],
      properties: {
        cwd:       { type: "string",  description: "Repository path" },
        message:   { type: "string",  description: "Commit message" },
        all:       { type: "boolean", description: "Automatically stage modified/deleted tracked files (-a)" },
        amend:     { type: "boolean", description: "Amend the last commit" },
        allow_empty: { type: "boolean", description: "Allow a commit with no changes" },
        author:    { type: "string",  description: "Override author in 'Name <email>' format" },
      },
    },
    async handler({ cwd, message, all, amend, allow_empty, author }) {
      const args = ["commit", "-m", message];
      if (all)         args.push("-a");
      if (amend)       args.push("--amend");
      if (allow_empty) args.push("--allow-empty");
      if (author)      args.push("--author", author);
      const res = await git(args, cwd);
      return ok(joinOut(res));
    },
  },

  // ── Log & show ───────────────────────────────────────────────────────────
  {
    name: "git_log",
    description: "Show the commit history.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:      { type: "string",  description: "Repository path" },
        max_count:{ type: "number",  description: "Maximum number of commits to show (default 20)" },
        oneline:  { type: "boolean", description: "One line per commit" },
        graph:    { type: "boolean", description: "Show ASCII branch graph" },
        all:      { type: "boolean", description: "Include all refs (--all)" },
        author:   { type: "string",  description: "Filter by author pattern" },
        since:    { type: "string",  description: "Show commits more recent than a date (e.g. '2 weeks ago')" },
        until:    { type: "string",  description: "Show commits older than a date" },
        paths:    { type: "array",   items: { type: "string" }, description: "Limit to these paths" },
        ref:      { type: "string",  description: "Start from this ref (default: HEAD)" },
      },
    },
    async handler({ cwd, max_count = 20, oneline, graph, all, author, since, until, paths, ref }) {
      const args = ["log", `--max-count=${max_count}`];
      if (oneline) args.push("--oneline");
      if (graph)   args.push("--graph");
      if (all)     args.push("--all");
      if (author)  args.push(`--author=${author}`);
      if (since)   args.push(`--since=${since}`);
      if (until)   args.push(`--until=${until}`);
      if (ref)     args.push(ref);
      if (paths?.length) args.push("--", ...paths);
      const res = await git(args, cwd);
      return ok(joinOut(res));
    },
  },
  {
    name: "git_show",
    description: "Show details of a commit, tag, or tree object.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:    { type: "string", description: "Repository path" },
        object: { type: "string", description: "Commit hash, tag, or ref (default: HEAD)" },
        stat:   { type: "boolean", description: "Show diffstat only" },
      },
    },
    async handler({ cwd, object = "HEAD", stat }) {
      const args = ["show", object];
      if (stat) args.push("--stat");
      const res = await git(args, cwd);
      return ok(joinOut(res));
    },
  },

  // ── Branches ─────────────────────────────────────────────────────────────
  {
    name: "git_branch",
    description: "List, create, rename, or delete branches.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:        { type: "string",  description: "Repository path" },
        list:       { type: "boolean", description: "List branches (default if no other action)" },
        all:        { type: "boolean", description: "List local + remote branches" },
        remotes:    { type: "boolean", description: "List remote-tracking branches" },
        create:     { type: "string",  description: "Name of new branch to create" },
        start_point:{ type: "string",  description: "Start point for the new branch" },
        rename:     { type: "object",
                      description: "Rename a branch: { old: string, new: string }",
                      properties: { old: { type: "string" }, new: { type: "string" } },
                      required: ["old","new"] },
        delete:     { type: "string",  description: "Branch name to delete" },
        force_delete: { type: "boolean", description: "Force-delete even if not merged (-D)" },
      },
    },
    async handler({ cwd, list, all, remotes, create, start_point, rename, delete: del, force_delete }) {
      if (create) {
        const args = ["branch", create];
        if (start_point) args.push(start_point);
        return ok(joinOut(await git(args, cwd)));
      }
      if (rename) {
        return ok(joinOut(await git(["branch", "-m", rename.old, rename.new], cwd)));
      }
      if (del) {
        const flag = force_delete ? "-D" : "-d";
        return ok(joinOut(await git(["branch", flag, del], cwd)));
      }
      // list
      const args = ["branch"];
      if (all)     args.push("-a");
      if (remotes) args.push("-r");
      return ok(joinOut(await git(args, cwd)));
    },
  },
  {
    name: "git_checkout",
    description: "Switch branches or restore working-tree files.",
    inputSchema: {
      type: "object",
      required: ["ref"],
      properties: {
        cwd:    { type: "string",  description: "Repository path" },
        ref:    { type: "string",  description: "Branch name, commit hash, or tag to check out" },
        create: { type: "boolean", description: "Create the branch if it doesn't exist (-b)" },
        paths:  { type: "array",   items: { type: "string" }, description: "Checkout only these paths" },
      },
    },
    async handler({ cwd, ref, create, paths }) {
      const args = ["checkout"];
      if (create) args.push("-b");
      args.push(ref);
      if (paths?.length) args.push("--", ...paths);
      const res = await git(args, cwd);
      return ok(joinOut(res));
    },
  },
  {
    name: "git_switch",
    description: "Switch branches (modern replacement for git checkout for branch switching).",
    inputSchema: {
      type: "object",
      required: ["branch"],
      properties: {
        cwd:    { type: "string",  description: "Repository path" },
        branch: { type: "string",  description: "Branch to switch to" },
        create: { type: "boolean", description: "Create the branch (-c)" },
        start_point: { type: "string", description: "Start point when creating a new branch" },
      },
    },
    async handler({ cwd, branch, create, start_point }) {
      const args = ["switch"];
      if (create) args.push("-c");
      args.push(branch);
      if (start_point) args.push(start_point);
      const res = await git(args, cwd);
      return ok(joinOut(res));
    },
  },

  // ── Merge & rebase ────────────────────────────────────────────────────────
  {
    name: "git_merge",
    description: "Merge one or more branches into the current branch.",
    inputSchema: {
      type: "object",
      required: ["branches"],
      properties: {
        cwd:       { type: "string",  description: "Repository path" },
        branches:  { type: "array",   items: { type: "string" }, description: "Branches to merge" },
        no_ff:     { type: "boolean", description: "Always create a merge commit (--no-ff)" },
        squash:    { type: "boolean", description: "Squash commits into one" },
        message:   { type: "string",  description: "Merge commit message" },
        abort:     { type: "boolean", description: "Abort an in-progress merge" },
        continue_: { type: "boolean", description: "Continue after resolving conflicts" },
      },
    },
    async handler({ cwd, branches, no_ff, squash, message, abort, continue_ }) {
      if (abort)     return ok(joinOut(await git(["merge", "--abort"], cwd)));
      if (continue_) return ok(joinOut(await git(["merge", "--continue"], cwd)));
      const args = ["merge"];
      if (no_ff)   args.push("--no-ff");
      if (squash)  args.push("--squash");
      if (message) args.push("-m", message);
      args.push(...branches);
      return ok(joinOut(await git(args, cwd)));
    },
  },
  {
    name: "git_rebase",
    description: "Rebase the current branch onto another.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:       { type: "string",  description: "Repository path" },
        onto:      { type: "string",  description: "The upstream branch to rebase onto" },
        abort:     { type: "boolean", description: "Abort an in-progress rebase" },
        continue_: { type: "boolean", description: "Continue after resolving conflicts" },
        skip:      { type: "boolean", description: "Skip the current patch" },
      },
    },
    async handler({ cwd, onto, abort, continue_, skip }) {
      if (abort)     return ok(joinOut(await git(["rebase", "--abort"],    cwd)));
      if (continue_) return ok(joinOut(await git(["rebase", "--continue"], cwd)));
      if (skip)      return ok(joinOut(await git(["rebase", "--skip"],     cwd)));
      const args = ["rebase"];
      if (onto) args.push(onto);
      return ok(joinOut(await git(args, cwd)));
    },
  },

  // ── Remote operations ─────────────────────────────────────────────────────
  {
    name: "git_remote",
    description: "Manage remote connections (list, add, remove, rename).",
    inputSchema: {
      type: "object",
      properties: {
        cwd:     { type: "string", description: "Repository path" },
        list:    { type: "boolean", description: "List remotes (default)" },
        verbose: { type: "boolean", description: "Show URLs" },
        add:     { type: "object", description: "Add a remote: { name, url }",
                   properties: { name: { type: "string" }, url: { type: "string" } }, required: ["name","url"] },
        remove:  { type: "string", description: "Remote name to remove" },
        rename:  { type: "object", description: "Rename a remote: { old, new }",
                   properties: { old: { type: "string" }, new: { type: "string" } }, required: ["old","new"] },
        set_url: { type: "object", description: "Change remote URL: { name, url }",
                   properties: { name: { type: "string" }, url: { type: "string" } }, required: ["name","url"] },
      },
    },
    async handler({ cwd, list, verbose, add, remove, rename, set_url }) {
      if (add)     return ok(joinOut(await git(["remote", "add", add.name, add.url], cwd)));
      if (remove)  return ok(joinOut(await git(["remote", "remove", remove], cwd)));
      if (rename)  return ok(joinOut(await git(["remote", "rename", rename.old, rename.new], cwd)));
      if (set_url) return ok(joinOut(await git(["remote", "set-url", set_url.name, set_url.url], cwd)));
      const args = ["remote"];
      if (verbose) args.push("-v");
      return ok(joinOut(await git(args, cwd)));
    },
  },
  {
    name: "git_fetch",
    description: "Download objects and refs from a remote.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:     { type: "string",  description: "Repository path" },
        remote:  { type: "string",  description: "Remote name (default: origin)" },
        branch:  { type: "string",  description: "Specific branch to fetch" },
        all:     { type: "boolean", description: "Fetch all remotes" },
        prune:   { type: "boolean", description: "Remove remote-tracking refs that no longer exist" },
        tags:    { type: "boolean", description: "Fetch all tags" },
      },
    },
    async handler({ cwd, remote = "origin", branch, all, prune, tags }) {
      const args = ["fetch"];
      if (all)   args.push("--all");
      if (prune) args.push("--prune");
      if (tags)  args.push("--tags");
      if (!all)  args.push(remote);
      if (branch) args.push(branch);
      return ok(joinOut(await git(args, cwd)));
    },
  },
  {
    name: "git_pull",
    description: "Fetch from a remote and merge (or rebase) into the current branch.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:     { type: "string",  description: "Repository path" },
        remote:  { type: "string",  description: "Remote name (default: origin)" },
        branch:  { type: "string",  description: "Remote branch to pull" },
        rebase:  { type: "boolean", description: "Rebase instead of merge (--rebase)" },
        ff_only: { type: "boolean", description: "Only fast-forward (--ff-only)" },
      },
    },
    async handler({ cwd, remote = "origin", branch, rebase, ff_only }) {
      const args = ["pull"];
      if (rebase)  args.push("--rebase");
      if (ff_only) args.push("--ff-only");
      args.push(remote);
      if (branch) args.push(branch);
      return ok(joinOut(await git(args, cwd)));
    },
  },
  {
    name: "git_push",
    description: "Update remote refs along with associated objects.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:         { type: "string",  description: "Repository path" },
        remote:      { type: "string",  description: "Remote name (default: origin)" },
        branch:      { type: "string",  description: "Local branch to push" },
        set_upstream:{ type: "boolean", description: "Set upstream tracking (-u)" },
        force:       { type: "boolean", description: "Force push (--force)" },
        force_with_lease: { type: "boolean", description: "Safer force push (--force-with-lease)" },
        delete:      { type: "string",  description: "Delete this remote branch" },
        tags:        { type: "boolean", description: "Push all tags" },
      },
    },
    async handler({ cwd, remote = "origin", branch, set_upstream, force, force_with_lease, delete: del, tags }) {
      const args = ["push"];
      if (set_upstream)   args.push("-u");
      if (force)          args.push("--force");
      if (force_with_lease) args.push("--force-with-lease");
      if (tags)           args.push("--tags");
      args.push(remote);
      if (del)    args.push("--delete", del);
      else if (branch) args.push(branch);
      return ok(joinOut(await git(args, cwd)));
    },
  },

  // ── Stash ─────────────────────────────────────────────────────────────────
  {
    name: "git_stash",
    description: "Stash (shelve) or restore dirty working-directory changes.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:      { type: "string",  description: "Repository path" },
        action:   { type: "string",
                    enum: ["push", "pop", "apply", "list", "drop", "clear", "show"],
                    description: "Stash sub-command (default: push)" },
        message:  { type: "string",  description: "Message for stash push" },
        index:    { type: "number",  description: "Stash index for pop/apply/drop/show (0 = latest)" },
        include_untracked: { type: "boolean", description: "Include untracked files in stash push (-u)" },
      },
    },
    async handler({ cwd, action = "push", message, index, include_untracked }) {
      const stashRef = index != null ? `stash@{${index}}` : null;
      switch (action) {
        case "list":  return ok(joinOut(await git(["stash", "list"], cwd)));
        case "clear": return ok(joinOut(await git(["stash", "clear"], cwd)));
        case "show": {
          const a = ["stash", "show", "-p"];
          if (stashRef) a.push(stashRef);
          return ok(joinOut(await git(a, cwd)));
        }
        case "pop":
        case "apply": {
          const a = ["stash", action];
          if (stashRef) a.push(stashRef);
          return ok(joinOut(await git(a, cwd)));
        }
        case "drop": {
          const a = ["stash", "drop"];
          if (stashRef) a.push(stashRef);
          return ok(joinOut(await git(a, cwd)));
        }
        default: { // push
          const a = ["stash", "push"];
          if (include_untracked) a.push("-u");
          if (message) a.push("-m", message);
          return ok(joinOut(await git(a, cwd)));
        }
      }
    },
  },

  // ── Tags ──────────────────────────────────────────────────────────────────
  {
    name: "git_tag",
    description: "List, create, or delete tags.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:      { type: "string",  description: "Repository path" },
        list:     { type: "boolean", description: "List tags (default)" },
        create:   { type: "string",  description: "Tag name to create" },
        annotate: { type: "boolean", description: "Create an annotated tag" },
        message:  { type: "string",  description: "Tag message (implies annotated)" },
        ref:      { type: "string",  description: "Commit to tag (default: HEAD)" },
        delete:   { type: "string",  description: "Tag name to delete" },
      },
    },
    async handler({ cwd, list, create, annotate, message, ref, delete: del }) {
      if (del) return ok(joinOut(await git(["tag", "-d", del], cwd)));
      if (create) {
        const a = ["tag"];
        if (annotate || message) a.push("-a");
        if (message) a.push("-m", message);
        a.push(create);
        if (ref) a.push(ref);
        return ok(joinOut(await git(a, cwd)));
      }
      return ok(joinOut(await git(["tag", "--list"], cwd)));
    },
  },

  // ── Reset & revert ────────────────────────────────────────────────────────
  {
    name: "git_reset",
    description: "Reset current HEAD to a specified state.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:    { type: "string", description: "Repository path" },
        commit: { type: "string", description: "Commit to reset to (default: HEAD)" },
        mode:   { type: "string", enum: ["soft", "mixed", "hard", "merge", "keep"],
                  description: "Reset mode (default: mixed)" },
        paths:  { type: "array",  items: { type: "string" }, description: "Limit reset to paths (implies mixed)" },
      },
    },
    async handler({ cwd, commit = "HEAD", mode = "mixed", paths }) {
      if (paths?.length) {
        return ok(joinOut(await git(["reset", commit, "--", ...paths], cwd)));
      }
      return ok(joinOut(await git(["reset", `--${mode}`, commit], cwd)));
    },
  },
  {
    name: "git_revert",
    description: "Create new commits that undo previous commits.",
    inputSchema: {
      type: "object",
      required: ["commits"],
      properties: {
        cwd:      { type: "string", description: "Repository path" },
        commits:  { type: "array",  items: { type: "string" }, description: "Commits to revert (in order)" },
        no_commit:{ type: "boolean", description: "Stage changes without committing (--no-commit)" },
      },
    },
    async handler({ cwd, commits, no_commit }) {
      const args = ["revert"];
      if (no_commit) args.push("--no-commit");
      args.push(...commits);
      return ok(joinOut(await git(args, cwd)));
    },
  },

  // ── Cherry-pick ───────────────────────────────────────────────────────────
  {
    name: "git_cherry_pick",
    description: "Apply changes from specific commits onto the current branch.",
    inputSchema: {
      type: "object",
      required: ["commits"],
      properties: {
        cwd:       { type: "string",  description: "Repository path" },
        commits:   { type: "array",   items: { type: "string" }, description: "Commits to cherry-pick" },
        no_commit: { type: "boolean", description: "Stage changes without committing (-n)" },
        abort:     { type: "boolean", description: "Abort an in-progress cherry-pick" },
        continue_: { type: "boolean", description: "Continue after resolving conflicts" },
      },
    },
    async handler({ cwd, commits, no_commit, abort, continue_ }) {
      if (abort)     return ok(joinOut(await git(["cherry-pick", "--abort"],    cwd)));
      if (continue_) return ok(joinOut(await git(["cherry-pick", "--continue"], cwd)));
      const args = ["cherry-pick"];
      if (no_commit) args.push("-n");
      args.push(...commits);
      return ok(joinOut(await git(args, cwd)));
    },
  },

  // ── Inspection ────────────────────────────────────────────────────────────
  {
    name: "git_blame",
    description: "Show which commit and author last modified each line of a file.",
    inputSchema: {
      type: "object",
      required: ["file"],
      properties: {
        cwd:   { type: "string",  description: "Repository path" },
        file:  { type: "string",  description: "File to annotate" },
        line_start: { type: "number", description: "Start line" },
        line_end:   { type: "number", description: "End line" },
        rev:   { type: "string",  description: "Blame at this revision" },
      },
    },
    async handler({ cwd, file, line_start, line_end, rev }) {
      const args = ["blame"];
      if (line_start && line_end) args.push(`-L${line_start},${line_end}`);
      else if (line_start) args.push(`-L${line_start}`);
      if (rev) args.push(rev);
      args.push("--", file);
      return ok(joinOut(await git(args, cwd)));
    },
  },
  {
    name: "git_grep",
    description: "Search for a pattern in tracked files.",
    inputSchema: {
      type: "object",
      required: ["pattern"],
      properties: {
        cwd:         { type: "string",  description: "Repository path" },
        pattern:     { type: "string",  description: "Search pattern (basic regex)" },
        ignore_case: { type: "boolean", description: "Case-insensitive search" },
        line_number: { type: "boolean", description: "Show line numbers" },
        count:       { type: "boolean", description: "Show match counts only" },
        paths:       { type: "array",   items: { type: "string" }, description: "Limit to these paths/globs" },
        rev:         { type: "string",  description: "Search in this tree-ish" },
      },
    },
    async handler({ cwd, pattern, ignore_case, line_number, count, paths, rev }) {
      const args = ["grep"];
      if (ignore_case)  args.push("-i");
      if (line_number)  args.push("-n");
      if (count)        args.push("-c");
      args.push(pattern);
      if (rev) args.push(rev);
      if (paths?.length) args.push("--", ...paths);
      return ok(joinOut(await git(args, cwd)));
    },
  },
  {
    name: "git_ls_files",
    description: "List files in the index (tracked files).",
    inputSchema: {
      type: "object",
      properties: {
        cwd:      { type: "string",  description: "Repository path" },
        others:   { type: "boolean", description: "Show untracked files" },
        modified: { type: "boolean", description: "Show modified files" },
        deleted:  { type: "boolean", description: "Show deleted files" },
        cached:   { type: "boolean", description: "Show cached/staged files (default)" },
        paths:    { type: "array",   items: { type: "string" }, description: "Limit to these paths" },
      },
    },
    async handler({ cwd, others, modified, deleted, cached, paths }) {
      const args = ["ls-files"];
      if (others)   args.push("-o");
      if (modified) args.push("-m");
      if (deleted)  args.push("-d");
      if (cached || (!others && !modified && !deleted)) args.push("-c");
      if (paths?.length) args.push("--", ...paths);
      return ok(joinOut(await git(args, cwd)));
    },
  },
  {
    name: "git_shortlog",
    description: "Summarise git log output — commits grouped by author.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:     { type: "string",  description: "Repository path" },
        summary: { type: "boolean", description: "Show only commit count per author (-s)" },
        numbered:{ type: "boolean", description: "Sort by number of commits (-n)" },
        email:   { type: "boolean", description: "Show email addresses (-e)" },
        ref:     { type: "string",  description: "Ref range (e.g. 'v1.0..HEAD')" },
      },
    },
    async handler({ cwd, summary, numbered, email, ref }) {
      const args = ["shortlog"];
      if (summary)  args.push("-s");
      if (numbered) args.push("-n");
      if (email)    args.push("-e");
      if (ref) args.push(ref);
      return ok(joinOut(await git(args, cwd)));
    },
  },

  // ── Config ────────────────────────────────────────────────────────────────
  {
    name: "git_config",
    description: "Read or write git configuration values.",
    inputSchema: {
      type: "object",
      properties: {
        cwd:    { type: "string", description: "Repository path (for local config)" },
        get:    { type: "string", description: "Config key to read (e.g. 'user.name')" },
        set:    { type: "object", description: "Key-value pair to set: { key, value }",
                  properties: { key: { type: "string" }, value: { type: "string" } }, required: ["key","value"] },
        list:   { type: "boolean", description: "List all config entries" },
        global: { type: "boolean", description: "Use global config instead of local" },
        unset:  { type: "string",  description: "Config key to remove" },
      },
    },
    async handler({ cwd, get, set, list, global: isGlobal, unset }) {
      const scope = isGlobal ? ["--global"] : ["--local"];
      if (list)  return ok(joinOut(await git(["config", ...scope, "--list"], cwd)));
      if (get)   return ok(joinOut(await git(["config", ...scope, "--get", get], cwd)));
      if (unset) return ok(joinOut(await git(["config", ...scope, "--unset", unset], cwd)));
      if (set)   return ok(joinOut(await git(["config", ...scope, set.key, set.value], cwd)));
      return ok("Specify get, set, list, or unset.");
    },
  },
];

// ─── MCP Server ──────────────────────────────────────────────────────────────

const server = new Server(
  { name: "git-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS.map(({ name, description, inputSchema }) => ({
    name,
    description,
    inputSchema,
  })),
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const tool = TOOLS.find((t) => t.name === request.params.name);
  if (!tool) {
    return {
      content: [{ type: "text", text: `Unknown tool: ${request.params.name}` }],
      isError: true,
    };
  }
  try {
    return await tool.handler(request.params.arguments ?? {});
  } catch (err) {
    return {
      content: [{ type: "text", text: `Error: ${err.message}` }],
      isError: true,
    };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
