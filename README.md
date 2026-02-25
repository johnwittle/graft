# graft

A conversation harness for the Anthropic API, designed for context continuity and persistent conversations with Claude.

## Features

- **Persistent conversations**: Save, load, and manage multiple conversation threads
- **Full context preservation**: Imports from claude.ai exports (via socketteer) with thinking blocks intact
- **Prompt caching**: Configurable TTL (5m/1h) to reduce costs on long conversations
- **Web search**: Optional integration with Anthropic's web search tool
- **Readable transcripts**: View conversation history in a pager, export to text files

## Installation
```bash
pip install -e .
```

Or copy `graft.py` directly to somewhere in your `$PATH`.

Requires an Anthropic API key in one of:
- `./.env`
- `~/.env`
- `~/.graft/.env`

## Usage
```bash
graft                    # Start new or select existing conversation
graft myconversation     # Load a saved conversation by name
graft --import file.json # Import from socketteer export
graft --list             # List saved conversations
```

### Commands

During a conversation, commands start with `/`:

| Command | Description |
|---------|-------------|
| `/save [name]` | Save conversation |
| `/load <name>` | Load a different conversation |
| `/list` | Show saved conversations |
| `/read` | View full transcript in pager |
| `/export [file]` | Export transcript to text file |
| `/delete <name>` | Delete a saved conversation |
| `/cache on\|off\|5m\|1h` | Control prompt caching |
| `/web on\|off` | Toggle web search |
| `/model [name]` | Show or switch model |
| `/tools [path]` | Enable file tools for a directory |
| `/shell on\|off` | Toggle shell command execution |
| `/tokens` | Show token estimate |
| `/stats` | Show session statistics |
| `/help` | Show all commands |
| `/quit` | Exit |

## Configuration

Settings are stored in `~/.graft/config.toml`:
```toml
default_model = "claude-opus-4-5-20251101"
cache_ttl = "5m"
editing_mode = "vi"  # or "emacs"
max_tokens = 8192
web_search = false
```

Conversations are saved in `~/.graft/conversations/`.

## Importing from claude.ai

Use the [socketteer Claude Conversation Exporter](https://github.com/socketteer/Claude-Conversation-Exporter) browser extension to export conversations from claude.ai, then:
```bash
graft --import "My Conversation.json" my-conversation
```

By default, thinking blocks and tool use are preserved. Use `--no-thinking` or `--no-tool-use` to exclude them.

## License

MIT

## Claude Code Integration

The `bin/` directory contains utilities for integrating with Claude Code while preserving session history:

### claude-sub

A wrapper for Claude Code that automatically archives sessions:

```bash
claude-sub "task description" "prompt for claude code"
```

Sessions are archived to `~/.claude-archive/` with timestamps and task descriptions.

### claude-session-to-graft

Convert archived Claude Code sessions to graft format:

```bash
claude-session-to-graft ~/.claude-archive/raw/session.jsonl output.json
# Then load in graft:
cp output.json ~/.graft/conversations/
graft output
```

### claude-archive-list

List archived Claude Code sessions:

```bash
claude-archive-list          # Summary view
claude-archive-list --full   # Full session log
```

### Setup

Add the bin directory to your PATH, or symlink the tools:

```bash
# Option 1: Add to PATH in ~/.bashrc
export PATH="$PATH:/path/to/graft/bin"

# Option 2: Symlink to existing bin directory
ln -s /path/to/graft/bin/claude-* ~/bin/
```

Create the archive directory:

```bash
mkdir -p ~/.claude-archive/raw
```

## Multi-Environment Setup

### Sharing Claude Code credentials across machines

If you have multiple environments (servers, VMs, etc.) where you want to use `claude-sub` with your Max plan, you can copy the OAuth credentials rather than re-authenticating on each machine.

**First-time setup (on a machine with a browser):**

1. Install Claude Code: `npm install -g @anthropic-ai/claude-code`
2. Run `claude` and complete the OAuth flow in your browser
3. When it asks you to set up a project directory, you can exit (Ctrl+C) - the credentials are already saved
4. Your credentials are now in `~/.claude/.credentials.json`

**Copying to other environments:**

```bash
# On the target machine
mkdir -p ~/.claude
chmod 700 ~/.claude

# Copy the credentials file (via scp, rsync, etc.)
scp source-machine:~/.claude/.credentials.json ~/.claude/
chmod 600 ~/.claude/.credentials.json
```

The OAuth tokens will auto-refresh when Claude Code next runs. Make sure you're using the actual `claude` binary - Anthropic whitelists official clients for OAuth access.

**Note:** `claude-sub` automatically unsets `ANTHROPIC_API_KEY` before calling Claude Code, ensuring subagent calls bill against your Max plan rather than API credits.
