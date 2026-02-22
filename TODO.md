# Graft TODO

## Completed ✓

- ~~Extract socketteer import to separate file~~ → `bin/graft-import`
- ~~Fix tool interleaving in imports~~ → proper assistant/user message splitting
- ~~Update /thinking to use 'adaptive'~~ → fixed in graft.py
- ~~Remove old convert_socketteer_format from graft.py~~ → replaced with subprocess call to `bin/graft-import`

## Medium Priority

### Fix /compress early end_turn bug
The compression feature asks Claude to output "User:" and "Assistant:" markers, but Claude has strong instincts against outputting patterns that look like impersonating users, causing premature end_turn. Needs a different output format (JSON? different markers?).

### Token estimation consistency
`Conversation.token_estimate()` uses `// 2` ratio, but `/compress` uses `// 4`. Should unify or document why they differ.

## Low Priority / Backlog

### Tests
No test suite currently. Priority candidates:
- `bin/graft-import` (now the most complex piece)
- `Conversation.save()`/`load()` round-trips
- `_parse_compressed_transcript()`

### Type hints
Would help AI contributors and reduce type confusion bugs.

### Tool permission enforcement
`ToolExecutor.execute()` runs any tool by name regardless of `/shell` and `/tools` settings. Low priority since Claude respects the advertised tool list anyway (see "short programs" discussion in graft-manager transcript).

---

*Last updated: 2026-02-22*
