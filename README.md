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
