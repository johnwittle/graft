# Graft TODO

## High Priority

### Extract socketteer import to separate file
The `convert_socketteer_format()` function and related import logic has grown complex enough to warrant its own file (`bin/graft-import` or `bin/socketteer-to-graft`). This would:
- Keep graft.py cleaner
- Make the import logic easier to test and iterate on
- Parallel the `claude-session-to-graft` pattern

### Fix tool interleaving in imports
Currently, imported conversations batch all tool_uses together and all tool_results together, losing temporal structure. Should interleave properly:
```
# Current (wrong):
assistant: [thinking, tool_use_1, thinking, tool_use_2, text]
user: [tool_result_1, tool_result_2]

# Correct:
assistant: [thinking, tool_use_1]
user: [tool_result_1]
assistant: [thinking, tool_use_2]
user: [tool_result_2]
assistant: [text]
```
Requires matching `tool_use.id` with `tool_result.tool_use_id` and splitting at each pair.

### Update /thinking to use 'adaptive'
API now warns: `'thinking.type=enabled' is deprecated. Use 'thinking.type=adaptive' instead`

## Medium Priority

### Fix /compress early end_turn bug
The compression feature asks Claude to output "User:" and "Assistant:" markers, but Claude has strong instincts against outputting patterns that look like impersonating users, causing premature end_turn. Needs a different output format (JSON? different markers?).

### Token estimation consistency
`Conversation.token_estimate()` uses `// 2` ratio, but `/compress` uses `// 4`. Should unify or document why they differ.

## Low Priority / Backlog

### Tests
No test suite currently. Priority candidates:
- `Conversation.save()`/`load()` round-trips
- `convert_socketteer_format()` 
- `_parse_compressed_transcript()`

### Type hints
Would help AI contributors and reduce type confusion bugs.

### Tool permission enforcement
`ToolExecutor.execute()` runs any tool by name regardless of `/shell` and `/tools` settings. Low priority since Claude respects the advertised tool list anyway (see "short programs" discussion in graft-manager transcript).

---

*Last updated: 2026-02-21*
