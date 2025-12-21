# TUI Best Practices for Improved UX

Research compiled from modern TUI frameworks and CLI design patterns.

## Rendering & Performance

### Flicker-Free Rendering

Two critical techniques from [Textual framework development](https://www.textualize.io/blog/7-things-ive-learned-building-a-modern-tui-framework/):

1. **Overwrite, don't clear**: Never clear the screen then redraw. Clearing creates blank intermediate frames. Always overwrite existing content in place.

2. **Single write to stdout**: Batch all updates into a single `write()` call. Multiple writes risk partial updates becoming visible between frames.

### Terminal Emulator Considerations

Modern terminals (iTerm, Kitty, WezTerm, Warp) use hardware-accelerated rendering and can achieve up to 60 FPS. This makes smooth animations possible when optimized correctly.

Reference: [Algorithms for high performance terminal apps](https://textual.textualize.io/blog/2024/12/12/algorithms-for-high-performance-terminal-apps/)

### Async & Non-Blocking Updates

- Use async patterns to manage multiple widget updates without blocking
- Implement spatial mapping to quickly determine which widgets are visible
- Only render visible portions of large or complex UIs

## Animation Guidelines

### When to Animate

Not all animation is equal. From Textual's learnings:

| Animation Type | Recommendation |
|----------------|----------------|
| Smooth scrolling | Helpful - maintains user's place in content |
| Progress indicators | Essential - shows system is responsive |
| Transitions between states | Moderate - provides context |
| Decorative effects | Avoid - often perceived as gratuitous |

### Animation Controls

Always provide a mechanism to disable animations for users who prefer reduced motion. This is both an accessibility requirement and a user preference consideration.

## Progress & Loading Feedback

### Choosing the Right Indicator

From [Evil Martians CLI UX patterns](https://evilmartians.com/chronicles/cli-ux-best-practices-3-patterns-for-improving-progress-displays):

| Duration | Indicator Type | Details |
|----------|----------------|---------|
| < 3 seconds | Spinner | Indeterminate, shows activity |
| 3-10 seconds | X of Y pattern | "Processing 3/10 files" |
| > 10 seconds | Progress bar | Determinate with percentage |

### Essential Practices

1. **Never leave users staring at a blank cursor** - Always show status for operations that take time

2. **Clear indicators on completion** - Most libraries auto-clear, but verify this happens

3. **Handle terminal width** - Adapt progress bar width to terminal size, handle resizing gracefully

4. **Proper cleanup on interrupts** - Restore terminal state if user cancels (Ctrl+C)

5. **Use color meaningfully** - Green checkmarks for success, red for errors, but respect `--no-color` and `NO_COLOR` environment variable

### Technical Implementation

Progress bars work by printing `\r[===...]` without newline. The `\r` returns cursor to line start and overwrites previous value. On non-dynamic terminals, fall back to printing new lines at slower intervals.

## Keyboard Navigation

### Standard Key Bindings

Users expect consistent shortcuts across TUI applications:

| Key | Action |
|-----|--------|
| `q` / `Ctrl+C` | Quit |
| `?` | Toggle help |
| `↑` / `k` | Navigate up |
| `↓` / `j` | Navigate down |
| `Enter` | Select/confirm |
| `Tab` | Navigate between sections |
| `Esc` | Cancel/back |
| `/` | Search (vim convention) |

### Design Principles

- **Keyboard-first**: TUI users expect to never need a mouse
- **Discoverable shortcuts**: Show available keys in footer or help panel
- **Vim-style optional**: Many power users expect `hjkl` navigation
- **Consistent modifiers**: `Ctrl+` for system actions, plain keys for navigation

## Accessibility

### Core Requirements

From [Textual's accessibility features](https://realpython.com/python-textual/):

- Screen reader integration
- Monochrome mode option
- High-contrast themes
- Color-blind friendly palettes

### Design Guidelines

- Clear, logical menu structure
- Informative status messages
- Consistent navigation patterns
- Visible focus indicators

## Architecture Pattern: Model-View-Update

The [Elm architecture](https://taranveerbains.ca/blog/13-making-a-tui-with-go) (used by BubbleTea, Textual) provides clean separation:

| Component | Responsibility |
|-----------|----------------|
| **Model** | Application state (data structures) |
| **View** | Render state to terminal output |
| **Update** | Process input, update state, return new model |

This pattern ensures:
- Predictable state management
- Easy testing (pure functions)
- Clear data flow
- Separation of concerns

## Recommendations for Swarm TUI

Based on research, priority improvements:

### High Priority

1. **Batch screen updates** - Ensure all dashboard updates write to stdout in single call
2. **Add progress indicators** - Show spinner during agent spawning, progress bar during voting
3. **Keyboard shortcuts footer** - Always visible bar showing `s:stop` `x:delete` `q:quit` `?:help`
4. **Consistent key bindings** - Align with vim conventions (`j/k` for up/down)

### Medium Priority

5. **Status colors** - Green for completed, yellow for in-progress, red for failed
6. **Clear on completion** - Clean up spinners/progress when operations finish
7. **Terminal width handling** - Adapt layout to narrow terminals gracefully
8. **Disable animation option** - Respect user preference via config or env var

### Lower Priority

9. **Screen reader support** - Ensure critical info is accessible
10. **Help overlay** - `?` toggles full keyboard shortcut reference
11. **Smooth scrolling** - For long agent lists or diff views

## References

- [7 Things I've Learned Building a Modern TUI Framework](https://www.textualize.io/blog/7-things-ive-learned-building-a-modern-tui-framework/)
- [CLI UX Best Practices: 3 Patterns for Progress Displays](https://evilmartians.com/chronicles/cli-ux-best-practices-3-patterns-for-improving-progress-displays)
- [Algorithms for High Performance Terminal Apps](https://textual.textualize.io/blog/2024/12/12/algorithms-for-high-performance-terminal-apps/)
- [Python Textual: Build Beautiful UIs in Terminal](https://realpython.com/python-textual/)
- [Loading & Progress Indicators - UI Components](https://uxdesign.cc/loading-progress-indicators-ui-components-series-f4b1fc35339a)
- [awesome-tuis - GitHub](https://github.com/rothgar/awesome-tuis)
