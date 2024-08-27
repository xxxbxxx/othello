


othello/src
```zig othello/src/main.zig
// raylib-zig (c) Nikolas Wipper 2023
const std = @import("std");
const rl = @import("raylib");
const gui = @import("raygui");
const othello = @import("othello.zig");

const assert = std.debug.assert;

const Vec2 = rl.Vector2;

pub fn main() anyerror!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        std.testing.expect(gpa.deinit() == .ok) catch unreachable;
    }

    var frame_arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer frame_arena.deinit();

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    // Initialization
    //--------------------------------------------------------------------------------------
    const screenWidth = 1000;
    const screenHeight = 720;

    rl.setConfigFlags(.{ .vsync_hint = true, .msaa_4x_hint = true, .window_resizable = true });
    rl.initWindow(screenWidth, screenHeight, "othello");
    defer rl.closeWindow(); // Close window and OpenGL context

    rl.setTargetFPS(10); // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------

    // game state
    const State = struct {
        board: othello.Board,
        nextcol: othello.Color,
    };
    const init_state: State = .{ .board = othello.init_board, .nextcol = .black };

    var history = std.ArrayList(State).init(gpa.allocator());
    defer history.deinit();

    var game_state = init_state;
    try history.append(game_state);

    // ui state
    const gui_board_pos: Vec2 = .{ .x = 20, .y = 20 };
    const gui_board_size: f32 = 640;
    var game_over = false;
    var showhelpers = true;
    var ai_engine = [_]othello.Engine{ undefined, .greedy, .none };
    var gui_dropdowns_states = [_]bool{ false, false };

    // Main game loop
    while (!rl.windowShouldClose()) { // Detect window close button or ESC key
        _ = frame_arena.reset(.retain_capacity);
        const frame_alloc = frame_arena.allocator();

        const clicked_square: ?othello.Coord = sq: {
            if (game_over) break :sq null;
            const engine = ai_engine[@intFromEnum(game_state.nextcol)];
            if (engine == .none) {
                if (rl.isMouseButtonPressed(rl.MouseButton.mouse_button_left)) {
                    const p = Vec2.multiply(
                        Vec2.subtract(rl.getMousePosition(), gui_board_pos),
                        .{ .x = 8.0 / gui_board_size, .y = 8.0 / gui_board_size },
                    );
                    if (p.x >= 0 and p.x < 8 and p.y >= 0 and p.y < 8)
                        break :sq .{ @intFromFloat(p.x), @intFromFloat(p.y) };
                }
                break :sq null;
            } else {
                break :sq othello.computeBestMove(engine, game_state.board, game_state.nextcol, frame_alloc, random);
            }
        };

        if (clicked_square) |sq| play: {
            game_state.board = othello.playAt(game_state.board, sq, game_state.nextcol) catch break :play;
            game_state.nextcol = game_state.nextcol.next();

            if (!game_over) {
                const can_play = othello.computeValidSquares(game_state.board, game_state.nextcol) != 0;
                if (!can_play) {
                    game_state.nextcol = game_state.nextcol.next();
                    if (othello.computeValidSquares(game_state.board, game_state.nextcol) == 0)
                        game_over = true;
                }
            }

            try history.append(game_state);
        }

        // Draw
        //----------------------------------------------------------------------------------
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.init(0, 33, 66, 255));
        drawBoard(game_state.board, gui_board_pos, gui_board_size);
        if (showhelpers)
            drawHelper(game_state.board, game_state.nextcol, gui_board_pos, gui_board_size);

        rl.drawText("Next: ", 700, 50, 30, rl.Color.light_gray);
        if (!game_over)
            drawPawn(.{ .x = 815, .y = 65 }, 20, game_state.nextcol);

        const score = othello.computeScore(game_state.board);
        const scoretxt = try std.fmt.allocPrintZ(frame_alloc, "Score: {} - {}", .{ score.blacks, score.whites });
        rl.drawText(scoretxt, 700, 100, 30, rl.Color.light_gray);

        if (history.items.len > 2) {
            if (gui.guiButton(.{ .x = 700, .y = 150, .width = 100, .height = 20 }, "Undo") != 0) {
                history.items.len -= 2;
                game_state = history.getLast();
                game_over = false;
            }
        }
        if (gui.guiButton(.{ .x = 820, .y = 150, .width = 100, .height = 20 }, "Restart") != 0) {
            game_over = false;
            game_state = init_state;
            history.clearRetainingCapacity();
            try history.append(game_state);
        }

        _ = gui.guiCheckBox(.{ .x = 700, .y = 500, .width = 20, .height = 20 }, "show helpers", &showhelpers);

        rl.drawText("Black AI: ", 700, 525, 10, rl.Color.light_gray);
        rl.drawText("White AI: ", 700, 550, 10, rl.Color.light_gray);
        const ai_list = comptime list: {
            var t: [:0]const u8 = "";
            for (std.meta.tags(othello.Engine), 0..) |tag, i| {
                t = if (i == 0) @tagName(tag) else t ++ ";" ++ @tagName(tag);
            }
            break :list t;
        };
        if (gui.guiDropdownBox(.{ .x = 750, .y = 525, .width = 100, .height = 20 }, ai_list, @ptrCast(&ai_engine[@intFromEnum(othello.Color.black)]), gui_dropdowns_states[0]) != 0) gui_dropdowns_states[0] = !gui_dropdowns_states[0];
        if (gui.guiDropdownBox(.{ .x = 750, .y = 550, .width = 100, .height = 20 }, ai_list, @ptrCast(&ai_engine[@intFromEnum(othello.Color.white)]), gui_dropdowns_states[1]) != 0) gui_dropdowns_states[1] = !gui_dropdowns_states[1];

        if (game_over) {
            rl.drawText("Game Over!", 175, 200, 125, rl.Color.light_gray);
            if (gui.guiButton(.{ .x = 400, .y = 350, .width = 100, .height = 50 }, "Restart") != 0) {
                game_over = false;
                game_state = init_state;
                history.clearRetainingCapacity();
                try history.append(game_state);
            }
        }
    }
}

fn addmul(p: Vec2, k: f32, d: Vec2) Vec2 {
    return Vec2.add(p, Vec2.multiply(d, .{ .x = k, .y = k }));
}
fn drawBoard(b: othello.Board, pos: Vec2, size: f32) void {
    const rect: rl.Rectangle = .{ .x = pos.x, .y = pos.y, .width = size, .height = size };
    rl.drawRectangleRounded(rect, 0.1, 7, rl.Color.dark_green);

    {
        const pos0 = pos;
        const posw = Vec2.add(pos, .{ .x = size, .y = 0 });
        const posh = Vec2.add(pos, .{ .x = 0, .y = size });
        for (0..8) |i| {
            if (i == 0) continue;
            rl.drawLineEx(
                addmul(pos0, size / 8, .{ .x = @floatFromInt(i), .y = 0 }),
                addmul(posh, size / 8, .{ .x = @floatFromInt(i), .y = 0 }),
                2.0,
                rl.Color.black,
            );
            rl.drawLineEx(
                addmul(pos0, size / 8, .{ .y = @floatFromInt(i), .x = 0 }),
                addmul(posw, size / 8, .{ .y = @floatFromInt(i), .x = 0 }),
                2.0,
                rl.Color.black,
            );
        }
    }

    {
        const pos0 = Vec2.add(pos, .{ .x = size / 16, .y = size / 16 });

        for (0..8) |y| {
            for (0..8) |x| {
                drawPawn(addmul(pos0, size / 8, .{ .x = @floatFromInt(x), .y = @floatFromInt(y) }), size / 20, b.get(x, y));
            }
        }
    }

    rl.drawRectangleRoundedLinesEx(rect, 0.1, 7, 5.0, rl.Color.black);
}

fn drawPawn(pos: Vec2, radius: f32, col: othello.Color) void {
    const rgb = switch (col) {
        .empty => return,
        .white => rl.Color.beige,
        .black => rl.Color.dark_brown,
    };

    rl.drawCircleV(pos, radius, rgb);
    rl.drawCircleLinesV(pos, radius, rl.Color.black);
}

fn drawHelper(b: othello.Board, nextcol: othello.Color, pos: Vec2, size: f32) void {
    const pos0 = Vec2.add(pos, .{ .x = size / 16, .y = size / 16 });

    const valids = othello.computeValidSquares(b, nextcol);
    for (0..8) |y| {
        for (0..8) |x| {
            if (valids & othello.bitmask(x, y) == 0) continue;
            rl.drawCircleLinesV(addmul(pos0, size / 8, .{ .x = @floatFromInt(x), .y = @floatFromInt(y) }), size / 20, rl.Color.red);
        }
    }
}

// tests

fn playGame(seed: u32, alloc: std.mem.Allocator, engine: othello.Engine) othello.Score {
    var prng = std.Random.DefaultPrng.init(seed); //std.testing.random_seed);
    const random = prng.random();

    var board: othello.Board = othello.init_board;
    var nextcol: othello.Color = .black;

    var game_over = false;
    while (!game_over) {
        const pos = othello.computeBestMove(if (nextcol == .black) .random else engine, board, nextcol, alloc, random);
        board = othello.playAt(board, pos, nextcol) catch break;
        nextcol = nextcol.next();

        const can_play = othello.computeValidSquares(board, nextcol) != 0;
        if (!can_play) {
            nextcol = nextcol.next();
            if (othello.computeValidSquares(board, nextcol) == 0)
                game_over = true;
        }
    }
    return othello.computeScore(board);
}

test "random" {
    const score = playGame(4321, std.testing.allocator, .random);
    try std.testing.expectEqual(@as(u32, 32), score.whites);
    try std.testing.expectEqual(@as(u32, 32), score.blacks);
}

test "greedy" {
    const score = playGame(4321, std.testing.allocator, .greedy);
    try std.testing.expectEqual(@as(u32, 31), score.whites);
    try std.testing.expectEqual(@as(u32, 33), score.blacks);
}

test "multi steps" {
    const score = playGame(4321, std.testing.allocator, .five_steps);
    try std.testing.expectEqual(@as(u32, 57), score.whites);
    try std.testing.expectEqual(@as(u32, 4), score.blacks);
}

test "fuzz" {
    const input_bytes = std.testing.fuzzInput(.{});
    var input_idx: usize = 0;
    var board: othello.Board = othello.init_board;
    var nextcol: othello.Color = .black;

    var game_over = false;
    loop: while (!game_over) {
        const pos: othello.Coord = pos: {
            if (input_idx >= input_bytes.len) break :loop;
            const byte = input_bytes[input_idx];
            input_idx += 1;
            break :pos .{ @intCast(byte % 8), @intCast((byte / 8) % 8) };
        };
        board = othello.playAt(board, pos, nextcol) catch break;
        nextcol = nextcol.next();

        const can_play = othello.computeValidSquares(board, nextcol) != 0;
        if (!can_play) {
            nextcol = nextcol.next();
            if (othello.computeValidSquares(board, nextcol) == 0)
                game_over = true;
        }
    }

    try std.testing.expect(!game_over);
}
```
```zig othello/src/othello.zig
const std = @import("std");
const assert = std.debug.assert;

pub const Coord = [2]i8; //@Vector(2, i8); ..selfhosted
fn add(a: Coord, b: Coord) Coord {
    return .{ a[0] + b[0], a[1] + b[1] };
}
fn inBounds(p: Coord, comptime min: Coord, comptime max: Coord) bool {
    // @reduce(.And, @min(@max(p, V2i{ 0, 0 }), V2i{ 7, 7 }) == p))
    return @min(@max(p[0], min[0]), max[0]) == p[0] // x
    and @min(@max(p[1], min[1]), max[1]) == p[1]; // y
}

pub fn bitmask(x: anytype, y: anytype) u64 {
    assert(x >= 0 and y >= 0 and x < 8 and y < 8);
    return @as(u64, 1) << @intCast(y * 8 + x);
}

pub const Color = enum {
    empty,
    black,
    white,
    pub fn next(c: Color) Color {
        return switch (c) {
            .empty => .empty,
            .black => .white,
            .white => .black,
        };
    }
};
pub const Board = struct {
    occupied: u64,
    color: u64,

    pub fn get(b: @This(), x: anytype, y: anytype) Color {
        const bit: u64 = bitmask(x, y);
        if ((b.occupied & bit) == 0) return .empty;
        return if ((b.color & bit) == 0) .black else .white;
    }
    pub fn setBitmask(b: *@This(), bit: u64, c: Color) void {
        switch (c) {
            .empty => b.occupied &= ~bit,
            .black => {
                b.occupied |= bit;
                b.color &= ~bit;
            },
            .white => {
                b.occupied |= bit;
                b.color |= bit;
            },
        }
    }

    pub fn set(b: *@This(), x: anytype, y: anytype, c: Color) void {
        const bit: u64 = bitmask(x, y);
        setBitmask(b, bit, c);
    }
};

pub const init_board: Board = init: {
    var b: Board = .{ .color = 0, .occupied = 0 };
    b.set(3, 3, .white);
    b.set(4, 4, .white);
    b.set(3, 4, .black);
    b.set(4, 3, .black);
    break :init b;
};

const dirsteps: [8 * 8][8][]const u64 = blk: {
    @setEvalBranchQuota(10000);
    // nb: testé avec u8 ibit pltuot que u64 mask.  -> 1.22x plus lent

    const compass_dirs: []const Coord = &.{
        .{ 1, 0 }, .{ -1, 0 }, .{ 0, 1 },  .{ 0, -1 },
        .{ 1, 1 }, .{ -1, 1 }, .{ 1, -1 }, .{ -1, -1 },
    };
    var allsteps: [8 * 8][8][]const u64 = undefined;
    for (0..8) |y| {
        for (0..8) |x| {
            nextdir: for (compass_dirs, 0..) |d, idir| {
                var steps: []const u64 = &.{};
                var p: Coord = .{ @intCast(x), @intCast(y) };
                while (true) {
                    p = add(p, d);
                    if (!inBounds(p, Coord{ 0, 0 }, Coord{ 7, 7 })) {
                        allsteps[y * 8 + x][idir] = steps;
                        continue :nextdir;
                    }
                    steps = steps ++ .{bitmask(p[0], p[1])};
                }
            }
        }
    }
    break :blk allsteps;
};

pub fn computeValidSquares(b: Board, col: Color) u64 {
    assert(col != .empty);
    var valid: u64 = 0;
    for (&dirsteps, 0..) |dirs, index| {
        const bit0: u64 = @as(u64, 1) << @intCast(index);
        if ((b.occupied & bit0) != 0) continue;
        const ok: bool = loop: for (&dirs) |steps| {
            var has_oppo = false;
            for (steps) |bit| {
                if ((b.occupied & bit) == 0) continue :loop;
                const sq: Color = if ((b.color & bit) == 0) .black else .white;
                if (sq == col and !has_oppo) continue :loop;
                if (sq == col and has_oppo) break :loop true;
                has_oppo = true;
            }
        } else false;

        if (ok) {
            valid |= bit0;
        }
    }
    return valid;
}

pub fn playAt(b: Board, p: Coord, col: Color) !Board {
    assert(col != .empty);
    const valid = computeValidSquares(b, col);
    const bit0 = bitmask(p[0], p[1]);
    if (valid & bit0 == 0) return error.invalid;
    var b1 = b;
    b1.setBitmask(bit0, col);

    loop: for (&dirsteps[@intCast(p[1] * 8 + p[0])]) |steps| {
        for (steps) |bit| {
            if ((b.occupied & bit) == 0) continue :loop;
            const sq: Color = if ((b.color & bit) == 0) .black else .white;
            if (sq == col) break;
        } else continue :loop;

        for (steps) |bit| {
            assert((b.occupied & bit) != 0);
            const sq: Color = if ((b.color & bit) == 0) .black else .white;
            if (sq == col) continue :loop;
            b1.setBitmask(bit, col);
        }
        unreachable;
    }
    return b1;
}

pub const Score = struct { whites: u32, blacks: u32 };
pub fn computeScore(b: Board) Score {
    var nbw: u32 = 0;
    var nbb: u32 = 0;
    var bit: u64 = 1;
    for (0..64) |_| {
        nbb += @intFromBool((b.occupied & bit != 0) and (b.color & bit == 0));
        nbw += @intFromBool((b.occupied & bit != 0) and (b.color & bit != 0));
        bit <<= 1;
    }
    return .{ .whites = nbw, .blacks = nbb };
}

// bon pas d'état persistant pour l'instant
pub const Engine = enum(i32) {
    none,
    random,
    greedy,
    one_step,
    two_steps,
    five_steps,
};

const Context = struct {
    dico: std.AutoHashMap(Board, i32),
};

pub fn computeBestMove(engine: Engine, b: Board, col: Color, alloc: std.mem.Allocator, random: std.Random) Coord {
    var ctx: Context = .{
        .dico = std.AutoHashMap(Board, i32).init(alloc),
    };
    defer ctx.dico.deinit();

    return switch (engine) {
        .none => unreachable,
        .random => computeRandomMove(b, col, random),
        .greedy => computeGreedyMove(b, col, random).?.coord,
        .one_step => computeStepBestMove(b, col, random, &ctx, 1).?.coord,
        .two_steps => computeStepBestMove(b, col, random, &ctx, 2).?.coord,
        .five_steps => computeStepBestMove(b, col, random, &ctx, 5).?.coord,
    };
}

fn computeRandomMove(b: Board, col: Color, random: std.Random) Coord {
    const valids = computeValidSquares(b, col);
    while (true) {
        const index = random.intRangeAtMost(u6, 0, 63);
        const bit = @as(u64, 1) << index;
        if (valids & bit == 0) continue;
        return .{ index % 8, index / 8 };
    }
}

fn evaluation(b: Board, col: Color) i32 {
    const score = computeScore(b);
    const nb0: i64 = if (col == .white) score.whites else score.blacks;
    const nb1: i64 = if (col == .black) score.whites else score.blacks;
    const nbempty: i32 = @intCast(64 - (score.whites + score.blacks));
    const delta: i32 = @intCast(nb0 - nb1);
    return delta + nbempty; // on considère que tous les vides sont à nous (pour éviter de chercher à maximiser la diff plutot que finir la partie)
}

fn computeGreedyMove(b: Board, col: Color, random: std.Random) ?struct { coord: Coord, score: i32 } {
    const valids = computeValidSquares(b, col);

    var best: ?Coord = null;
    var best_score: i32 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            const bit = bitmask(x, y);
            if (valids & bit == 0) continue;

            const coord: Coord = .{ @intCast(x), @intCast(y) };
            const after = playAt(b, coord, col) catch unreachable;
            const delta = evaluation(after, col);
            if (best_score < delta or best == null) {
                best = coord;
                best_score = delta;
            } else if (best_score == delta and random.boolean()) {
                best = coord;
                best_score = delta;
            }
        }
    }
    return if (best) |coord| .{ .coord = coord, .score = best_score } else null;
}

fn computeStepBestMove(b: Board, col: Color, random: std.Random, ctx: *Context, lookahead: u32) ?struct { coord: Coord, score: i32 } {
    const valids = computeValidSquares(b, col);

    var best: ?Coord = null;
    var best_score: i32 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            const bit = bitmask(x, y);
            if (valids & bit == 0) continue;

            const coord: Coord = .{ @intCast(x), @intCast(y) };
            const after = playAt(b, coord, col) catch unreachable;

            const expected: i32 = score: {
                if (ctx.dico.get(after)) |v| break :score v;
                switch (lookahead) {
                    0 => {},
                    1 => {
                        if (computeGreedyMove(after, col.next(), random)) |res|
                            break :score -res.score;
                    },
                    else => {
                        if (computeStepBestMove(after, col.next(), random, ctx, lookahead - 1)) |res|
                            break :score -res.score;
                    },
                }
                break :score evaluation(after, col);
            };
            ctx.dico.put(after, expected) catch unreachable;

            if (best_score < expected or best == null) {
                best = coord;
                best_score = expected;
            } else if (best_score == expected and random.boolean()) {
                best = coord;
                best_score = expected;
            }
        }
    }
    return if (best) |coord| .{ .coord = coord, .score = best_score } else null;
}
```

improve evaluation function

To improve the evaluation function for the Othello game, we can incorporate some strategic elements that are known to be important in Othello. Here's an enhanced version of the evaluation function:

```zig
fn evaluation(b: Board, col: Color) i32 {
    const score = computeScore(b);
    const nb0: i32 = @intCast(if (col == .white) score.whites else score.blacks);
    const nb1: i32 = @intCast(if (col == .black) score.whites else score.blacks);
    const nbempty: i32 = @intCast(64 - (score.whites + score.blacks));

    // Piece difference
    var eval: i32 = (nb0 - nb1) * 100;

    // Corner occupancy
    eval += cornerOccupancy(b, col) * 1000;

    // Corner closeness
    eval += cornerCloseness(b, col) * 500;

    // Mobility (number of valid moves)
    const mobility = @popCount(computeValidSquares(b, col)) -
                     @popCount(computeValidSquares(b, col.next()));
    eval += mobility * 50;

    // Stability (edges and stable discs)
    eval += stabilityEvaluation(b, col) * 200;

    // End game considerations
    if (nbempty < 10) {
        eval += (nb0 - nb1) * 200; // Emphasize piece difference in endgame
    }

    return eval;
}

fn cornerOccupancy(b: Board, col: Color) i32 {
    var count: i32 = 0;
    const corners = [_]Coord{ .{0,0}, .{0,7}, .{7,0}, .{7,7} };
    for (corners) |c| {
        if (b.get(c[0], c[1]) == col) {
            count += 1;
        } else if (b.get(c[0], c[1]) == col.next()) {
            count -= 1;
        }
    }
    return count;
}

fn cornerCloseness(b: Board, col: Color) i32 {
    var count: i32 = 0;
    const near_corners = [_]Coord{
        .{0,1}, .{1,0}, .{1,1},
        .{0,6}, .{1,6}, .{1,7},
        .{6,0}, .{6,1}, .{7,1},
        .{6,6}, .{6,7}, .{7,6}
    };
    for (near_corners) |c| {
        if (b.get(c[0], c[1]) == col) {
            count -= 1;
        } else if (b.get(c[0], c[1]) == col.next()) {
            count += 1;
        }
    }
    return count;
}

fn stabilityEvaluation(b: Board, col: Color) i32 {
    var count: i32 = 0;
    // Check edges
    for (0..8) |i| {
        if (b.get(0, i) == col) count += 1;
        if (b.get(7, i) == col) count += 1;
        if (b.get(i, 0) == col) count += 1;
        if (b.get(i, 7) == col) count += 1;
    }
    // More complex stability analysis could be added here
    return count;
}
```

This improved evaluation function considers several important factors in Othello strategy:

1. Piece difference: Still considered but weighted.
2. Corner occupancy: Controlling corners is crucial in Othello.
3. Corner closeness: Penalizes moves near corners if not occupied.
4. Mobility: Rewards having more possible moves than the opponent.
5. Stability: Rewards controlling edges and stable discs.
6. Endgame considerations: Emphasizes piece difference more in the endgame.

The weights (multipliers) for each factor can be fine-tuned based on testing and performance. This evaluation function should provide a more nuanced and strategically sound assessment of board positions, leading to stronger play by the AI.
change board representation to be
struct {
    white: u64,
    black: u64,
Certainly! Let's modify the `Board` structure and update the related functions to work with this new representation. Here's how we can change the board representation and adjust the relevant parts of the code:

```zig
pub const Board = struct {
    white: u64,
    black: u64,

    pub fn get(b: @This(), x: anytype, y: anytype) Color {
        const bit: u64 = bitmask(x, y);
        if ((b.white & bit) != 0) return .white;
        if ((b.black & bit) != 0) return .black;
        return .empty;
    }

    pub fn setBitmask(b: *@This(), bit: u64, c: Color) void {
        switch (c) {
            .empty => {
                b.white &= ~bit;
                b.black &= ~bit;
            },
            .white => {
                b.white |= bit;
                b.black &= ~bit;
            },
            .black => {
                b.black |= bit;
                b.white &= ~bit;
            },
        }
    }

    pub fn set(b: *@This(), x: anytype, y: anytype, c: Color) void {
        const bit: u64 = bitmask(x, y);
        setBitmask(b, bit, c);
    }

    pub fn occupied(b: @This()) u64 {
        return b.white | b.black;
    }
};

pub const init_board: Board = init: {
    var b: Board = .{ .white = 0, .black = 0 };
    b.set(3, 3, .white);
    b.set(4, 4, .white);
    b.set(3, 4, .black);
    b.set(4, 3, .black);
    break :init b;
};

pub fn computeValidSquares(b: Board, col: Color) u64 {
    assert(col != .empty);
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    var valid: u64 = 0;

    for (&dirsteps, 0..) |dirs, index| {
        const bit0: u64 = @as(u64, 1) << @intCast(index);
        if ((b.occupied() & bit0) != 0) continue;
        const ok: bool = loop: for (&dirs) |steps| {
            var has_oppo = false;
            for (steps) |bit| {
                if ((own & bit) != 0 and !has_oppo) continue :loop;
                if ((own & bit) != 0 and has_oppo) break :loop true;
                if ((opp & bit) != 0) {
                    has_oppo = true;
                } else {
                    continue :loop;
                }
            }
        } else false;

        if (ok) {
            valid |= bit0;
        }
    }
    return valid;
}

pub fn playAt(b: Board, p: Coord, col: Color) !Board {
    assert(col != .empty);
    const valid = computeValidSquares(b, col);
    const bit0 = bitmask(p[0], p[1]);
    if (valid & bit0 == 0) return error.invalid;
    var b1 = b;
    b1.setBitmask(bit0, col);

    const own = if (col == .white) &b1.white else &b1.black;
    const opp = if (col == .white) &b1.black else &b1.white;

    loop: for (&dirsteps[@intCast(p[1] * 8 + p[0])]) |steps| {
        for (steps) |bit| {
            if ((b.occupied() & bit) == 0) continue :loop;
            if ((own.* & bit) != 0) break;
        } else continue :loop;

        for (steps) |bit| {
            if ((opp.* & bit) == 0) continue :loop;
            own.* |= bit;
            opp.* &= ~bit;
        }
    }
    return b1;
}

pub fn computeScore(b: Board) Score {
    return .{
        .whites = @popCount(b.white),
        .blacks = @popCount(b.black),
    };
}
```

This new representation using separate `white` and `black` bitboards offers several advantages:

1. It's more memory-efficient, as we're using two 64-bit integers instead of three.
2. It allows for faster bitwise operations in many cases.
3. It simplifies some operations, like getting the occupied squares (just OR the two bitboards).

The `computeValidSquares`, `playAt`, and `computeScore` functions have been updated to work with this new representation. The core logic remains the same, but the bitwise operations are now performed directly on the `white` and `black` fields.

Remember to update any other parts of your code that directly access the `Board` structure to use this new representation. This change should make your Othello engine more efficient and potentially faster.
ok so I've tweked a bit the code. here it is:
```zig othello/src/othello.zig
const std = @import("std");
const assert = std.debug.assert;

pub const Coord = [2]i8; //@Vector(2, i8); ..selfhosted
fn add(a: Coord, b: Coord) Coord {
    return .{ a[0] + b[0], a[1] + b[1] };
}
fn inBounds(p: Coord, comptime min: Coord, comptime max: Coord) bool {
    // @reduce(.And, @min(@max(p, V2i{ 0, 0 }), V2i{ 7, 7 }) == p))
    return @min(@max(p[0], min[0]), max[0]) == p[0] // x
    and @min(@max(p[1], min[1]), max[1]) == p[1]; // y
}

pub fn bitmask(x: anytype, y: anytype) u64 {
    assert(x >= 0 and y >= 0 and x < 8 and y < 8);
    return @as(u64, 1) << @intCast(y * 8 + x);
}

pub const Color = enum {
    empty,
    black,
    white,
    pub fn next(c: Color) Color {
        return switch (c) {
            .empty => .empty,
            .black => .white,
            .white => .black,
        };
    }
};
pub const Board = struct {
    white: u64,
    black: u64,

    pub fn get(b: @This(), x: anytype, y: anytype) Color {
        const bit: u64 = bitmask(x, y);
        if ((b.white & bit) != 0) return .white;
        if ((b.black & bit) != 0) return .black;
        return .empty;
    }

    pub fn setBitmask(b: *@This(), bit: u64, c: Color) void {
        switch (c) {
            .empty => {
                b.white &= ~bit;
                b.black &= ~bit;
            },
            .white => {
                b.white |= bit;
                b.black &= ~bit;
            },
            .black => {
                b.black |= bit;
                b.white &= ~bit;
            },
        }
    }

    pub fn set(b: *@This(), x: anytype, y: anytype, c: Color) void {
        const bit: u64 = bitmask(x, y);
        setBitmask(b, bit, c);
    }
};

pub const init_board: Board = init: {
    var b: Board = .{ .white = 0, .black = 0 };
    b.set(3, 3, .white);
    b.set(4, 4, .white);
    b.set(3, 4, .black);
    b.set(4, 3, .black);
    break :init b;
};

const dirsteps: [8 * 8][8][]const u64 = blk: {
    @setEvalBranchQuota(10000);
    // nb: testé avec u8 ibit pltuot que u64 mask.  -> 1.22x plus lent

    const compass_dirs: []const Coord = &.{
        .{ 1, 0 }, .{ -1, 0 }, .{ 0, 1 },  .{ 0, -1 },
        .{ 1, 1 }, .{ -1, 1 }, .{ 1, -1 }, .{ -1, -1 },
    };
    var allsteps: [8 * 8][8][]const u64 = undefined;
    for (0..8) |y| {
        for (0..8) |x| {
            nextdir: for (compass_dirs, 0..) |d, idir| {
                var steps: []const u64 = &.{};
                var p: Coord = .{ @intCast(x), @intCast(y) };
                while (true) {
                    p = add(p, d);
                    if (!inBounds(p, Coord{ 0, 0 }, Coord{ 7, 7 })) {
                        allsteps[y * 8 + x][idir] = steps;
                        continue :nextdir;
                    }
                    steps = steps ++ .{bitmask(p[0], p[1])};
                }
            }
        }
    }
    break :blk allsteps;
};

pub fn computeValidSquares(b: Board, col: Color) u64 {
    assert(col != .empty);
    const occupied = b.white | b.black;
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    var valid: u64 = 0;

    for (&dirsteps, 0..) |dirs, index| {
        const bit0: u64 = @as(u64, 1) << @intCast(index);
        if (occupied & bit0 != 0) continue;
        const ok: bool = loop: for (&dirs) |steps| {
            var has_oppo = false;
            for (steps) |bit| {
                if (own & bit != 0) {
                    if (has_oppo) break :loop true else continue :loop;
                }
                if (opp & bit != 0) {
                    has_oppo = true;
                } else {
                    continue :loop;
                }
            }
        } else false;

        if (ok) {
            valid |= bit0;
        }
    }
    return valid;
}

pub fn playAt(b: Board, p: Coord, col: Color) !Board {
    assert(col != .empty);
    const valid = computeValidSquares(b, col);
    const bit0 = bitmask(p[0], p[1]);
    if (valid & bit0 == 0) return error.invalid;
    var b1 = b;
    b1.setBitmask(bit0, col);

    const occupied = b.white | b.black;
    const own = if (col == .white) &b1.white else &b1.black;
    const opp = if (col == .white) &b1.black else &b1.white;

    loop: for (&dirsteps[@intCast(p[1] * 8 + p[0])]) |steps| {
        for (steps) |bit| {
            if (occupied & bit == 0) continue :loop;
            if (own.* & bit != 0) break;
        } else continue :loop;

        for (steps) |bit| {
            if (opp.* & bit == 0) continue :loop;
            own.* |= bit;
            opp.* &= ~bit;
        }
    }
    return b1;
}

pub const Score = struct { whites: u32, blacks: u32 };
pub fn computeScore(b: Board) Score {
    return .{
        .whites = @popCount(b.white),
        .blacks = @popCount(b.black),
    };
}

// bon pas d'état persistant pour l'instant
pub const Engine = enum(i32) {
    none,
    random,
    greedy,
    one_step,
    two_steps,
    five_steps,
};

const Context = struct {
    dico: std.AutoHashMap(Board, i32),
};

pub fn computeBestMove(engine: Engine, b: Board, col: Color, alloc: std.mem.Allocator, random: std.Random) Coord {
    var ctx: Context = .{
        .dico = std.AutoHashMap(Board, i32).init(alloc),
    };
    defer ctx.dico.deinit();

    return switch (engine) {
        .none => unreachable,
        .random => computeRandomMove(b, col, random),
        .greedy => computeGreedyMove(b, col, random).?.coord,
        .one_step => computeStepBestMove(b, col, random, &ctx, 1).?.coord,
        .two_steps => computeStepBestMove(b, col, random, &ctx, 2).?.coord,
        .five_steps => computeStepBestMove(b, col, random, &ctx, 5).?.coord,
    };
}

fn computeRandomMove(b: Board, col: Color, random: std.Random) Coord {
    const valids = computeValidSquares(b, col);
    while (true) {
        const index = random.intRangeAtMost(u6, 0, 63);
        const bit = @as(u64, 1) << index;
        if (valids & bit == 0) continue;
        return .{ index % 8, index / 8 };
    }
}

fn evaluation(b: Board, col: Color) i32 {
    const score = computeScore(b);
    const nb0: i32 = @intCast(if (col == .white) score.whites else score.blacks);
    const nb1: i32 = @intCast(if (col == .black) score.whites else score.blacks);
    const nbempty: i32 = @intCast(64 - (score.whites + score.blacks));

    // Piece difference
    var eval: i32 = (nb0 - nb1) * 100;

    // Corner occupancy
    eval += cornerOccupancy(b, col) * 1000;

    // Corner closeness
    eval += cornerCloseness(b, col) * 500;

    // Mobility (number of valid moves)
    const mobility = @as(i32, @popCount(computeValidSquares(b, col))) -
        @as(i32, @popCount(computeValidSquares(b, col.next())));
    eval += mobility * 50;

    // Stability (edges and stable discs)
    eval += stabilityEvaluation(b, col) * 200;

    // End game considerations
    if (nbempty < 10) {
        eval += (nb0 - nb1) * 200; // Emphasize piece difference in endgame
    }

    return eval;
}

fn cornerOccupancy(b: Board, col: Color) i32 {
    var count: i32 = 0;
    const corners = [_]Coord{ .{ 0, 0 }, .{ 0, 7 }, .{ 7, 0 }, .{ 7, 7 } };
    for (corners) |c| {
        if (b.get(c[0], c[1]) == col) {
            count += 1;
        } else if (b.get(c[0], c[1]) == col.next()) {
            count -= 1;
        }
    }
    return count;
}

fn cornerCloseness(b: Board, col: Color) i32 {
    var count: i32 = 0;
    const near_corners = [_]Coord{ .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 }, .{ 0, 6 }, .{ 1, 6 }, .{ 1, 7 }, .{ 6, 0 }, .{ 6, 1 }, .{ 7, 1 }, .{ 6, 6 }, .{ 6, 7 }, .{ 7, 6 } };
    for (near_corners) |c| {
        if (b.get(c[0], c[1]) == col) {
            count -= 1;
        } else if (b.get(c[0], c[1]) == col.next()) {
            count += 1;
        }
    }
    return count;
}

fn stabilityEvaluation(b: Board, col: Color) i32 {
    var count: i32 = 0;
    // Check edges
    for (0..8) |i| {
        if (b.get(0, i) == col) count += 1;
        if (b.get(7, i) == col) count += 1;
        if (b.get(i, 0) == col) count += 1;
        if (b.get(i, 7) == col) count += 1;
    }
    // More complex stability analysis could be added here
    return count;
}
fn computeGreedyMove(b: Board, col: Color, random: std.Random) ?struct { coord: Coord, score: i32 } {
    const valids = computeValidSquares(b, col);

    var best: ?Coord = null;
    var best_score: i32 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            const bit = bitmask(x, y);
            if (valids & bit == 0) continue;

            const coord: Coord = .{ @intCast(x), @intCast(y) };
            const after = playAt(b, coord, col) catch unreachable;
            const delta = evaluation(after, col);
            if (best_score < delta or best == null) {
                best = coord;
                best_score = delta;
            } else if (best_score == delta and random.boolean()) {
                best = coord;
                best_score = delta;
            }
        }
    }
    return if (best) |coord| .{ .coord = coord, .score = best_score } else null;
}

fn computeStepBestMove(b: Board, col: Color, random: std.Random, ctx: *Context, lookahead: u32) ?struct { coord: Coord, score: i32 } {
    const valids = computeValidSquares(b, col);

    var best: ?Coord = null;
    var best_score: i32 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            const bit = bitmask(x, y);
            if (valids & bit == 0) continue;

            const coord: Coord = .{ @intCast(x), @intCast(y) };
            const after = playAt(b, coord, col) catch unreachable;

            const expected: i32 = score: {
                if (ctx.dico.get(after)) |v| break :score v;
                switch (lookahead) {
                    0 => {},
                    1 => {
                        if (computeGreedyMove(after, col.next(), random)) |res|
                            break :score -res.score;
                    },
                    else => {
                        if (computeStepBestMove(after, col.next(), random, ctx, lookahead - 1)) |res|
                            break :score -res.score;
                    },
                }
                break :score evaluation(after, col);
            };
            ctx.dico.put(after, expected) catch unreachable;

            if (best_score < expected or best == null) {
                best = coord;
                best_score = expected;
            } else if (best_score == expected and random.boolean()) {
                best = coord;
                best_score = expected;
            }
        }
    }
    return if (best) |coord| .{ .coord = coord, .score = best_score } else null;
}
```

The changes you've made look good. You've successfully integrated the new board representation and updated the evaluation function as suggested. Here are a few observations and suggestions:

1. The `computeValidSquares` function has been optimized to work with the new board representation, which is great.

2. The `playAt` function has been updated correctly to work with the new board structure.

3. The evaluation function now includes the strategic elements we discussed, which should lead to stronger play.

4. The `computeGreedyMove` and `computeStepBestMove` functions have been updated to use the new evaluation function.

Here are a couple of suggestions for further improvement:

1. Consider adding a `occupied` method to the `Board` struct for cleaner code:

```zig
pub const Board = struct {
    white: u64,
    black: u64,

    // ... other methods ...

    pub fn occupied(self: @This()) u64 {
        return self.white | self.black;
    }
};
```

Then you can replace `b.white | b.black` with `b.occupied()` in your code.

2. You might want to add some weights to the evaluation function that can be easily tuned:

```zig
const EvalWeights = struct {
    piece_diff: i32 = 100,
    corner_occupancy: i32 = 1000,
    corner_closeness: i32 = 500,
    mobility: i32 = 50,
    stability: i32 = 200,
    endgame_piece_diff: i32 = 200,
};

const weights = EvalWeights{};

fn evaluation(b: Board, col: Color) i32 {
    // ... (rest of the function remains the same)

    var eval: i32 = (nb0 - nb1) * weights.piece_diff;
    eval += cornerOccupancy(b, col) * weights.corner_occupancy;
    eval += cornerCloseness(b, col) * weights.corner_closeness;
    eval += mobility * weights.mobility;
    eval += stabilityEvaluation(b, col) * weights.stability;

    if (nbempty < 10) {
        eval += (nb0 - nb1) * weights.endgame_piece_diff;
    }

    return eval;
}
```

This makes it easier to fine-tune the evaluation function by adjusting these weights.

3. Consider adding a simple time management system to your `computeStepBestMove` function. This could help prevent the AI from taking too long on a single move, especially in the endgame where the search tree can become very deep.

Overall, your implementation looks solid and should provide a strong Othello AI. The next step would be to test it against other implementations or human players and adjust the evaluation weights based on its performance.
I think it could be further optimized. for instance by using bitmasks to extract corner or edge pieces instead of looping.

You're absolutely right. Using bitmasks for operations like corner and edge piece evaluation can significantly improve performance. Let's optimize these parts of the code:

1. Corner Occupancy:

```zig
const CORNER_MASK: u64 = 0x8100000000000081; // Corners: a1, h1, a8, h8

fn cornerOccupancy(b: Board, col: Color) i32 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    const own_corners = @popCount(own & CORNER_MASK);
    const opp_corners = @popCount(opp & CORNER_MASK);
    return @as(i32, own_corners) - @as(i32, opp_corners);
}
```

2. Corner Closeness:

```zig
const X_SQUARE_MASK: u64 = 0x42C300000000C342; // X-squares adjacent to corners

fn cornerCloseness(b: Board, col: Color) i32 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    const own_x_squares = @popCount(own & X_SQUARE_MASK);
    const opp_x_squares = @popCount(opp & X_SQUARE_MASK);
    return @as(i32, opp_x_squares) - @as(i32, own_x_squares);
}
```

3. Stability Evaluation (Edges):

```zig
const EDGE_MASK: u64 = 0xFF818181818181FF; // All edge squares

fn stabilityEvaluation(b: Board, col: Color) i32 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    const own_edges = @popCount(own & EDGE_MASK);
    const opp_edges = @popCount(opp & EDGE_MASK);
    return @as(i32, own_edges) - @as(i32, opp_edges);
}
```

4. Mobility (already optimized, but let's make it consistent with the above style):

```zig
fn mobility(b: Board, col: Color) i32 {
    const own_mobility = @popCount(computeValidSquares(b, col));
    const opp_mobility = @popCount(computeValidSquares(b, col.next()));
    return @as(i32, own_mobility) - @as(i32, opp_mobility);
}
```

Now, let's update the evaluation function:

```zig
const EvalWeights = struct {
    piece_diff: i32 = 100,
    corner_occupancy: i32 = 1000,
    corner_closeness: i32 = 500,
    mobility: i32 = 50,
    stability: i32 = 200,
    endgame_piece_diff: i32 = 200,
};

const weights = EvalWeights{};

fn evaluation(b: Board, col: Color) i32 {
    const score = computeScore(b);
    const nb0: i32 = @intCast(if (col == .white) score.whites else score.blacks);
    const nb1: i32 = @intCast(if (col == .black) score.whites else score.blacks);
    const nbempty: i32 = @intCast(64 - (score.whites + score.blacks));

    var eval: i32 = (nb0 - nb1) * weights.piece_diff;
    eval += cornerOccupancy(b, col) * weights.corner_occupancy;
    eval += cornerCloseness(b, col) * weights.corner_closeness;
    eval += mobility(b, col) * weights.mobility;
    eval += stabilityEvaluation(b, col) * weights.stability;

    if (nbempty < 10) {
        eval += (nb0 - nb1) * weights.endgame_piece_diff;
    }

    return eval;
}
```

These optimizations should significantly improve the performance of your evaluation function. The use of bitmasks allows for very fast bitwise operations instead of loops, which is particularly efficient for 64-bit board representations.

Additionally, you might want to consider precomputing some of these masks (like valid move masks for each square) to further optimize the `computeValidSquares` function, but that would require a more extensive refactoring of the code.
I think this is too many functions, can you inline them into evaluation() by using zig const val = blk: {}; syntax?

Certainly! We can inline these functions into the `evaluation()` function using Zig's `const val = blk: {};` syntax. This approach can potentially improve performance by reducing function call overhead. Here's the optimized `evaluation()` function with inlined calculations:

```zig
const EvalWeights = struct {
    piece_diff: i32 = 100,
    corner_occupancy: i32 = 1000,
    corner_closeness: i32 = 500,
    mobility: i32 = 50,
    stability: i32 = 200,
    endgame_piece_diff: i32 = 200,
};

const weights = EvalWeights{};

fn evaluation(b: Board, col: Color) i32 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;

    const score = computeScore(b);
    const nb0: i32 = @intCast(if (col == .white) score.whites else score.blacks);
    const nb1: i32 = @intCast(if (col == .black) score.whites else score.blacks);
    const nbempty: i32 = @intCast(64 - (score.whites + score.blacks));

    var eval: i32 = (nb0 - nb1) * weights.piece_diff;

    // Corner occupancy
    const corner_occupancy = blk: {
        const CORNER_MASK: u64 = 0x8100000000000081; // Corners: a1, h1, a8, h8
        const own_corners = @popCount(own & CORNER_MASK);
        const opp_corners = @popCount(opp & CORNER_MASK);
        break :blk @as(i32, own_corners) - @as(i32, opp_corners);
    };
    eval += corner_occupancy * weights.corner_occupancy;

    // Corner closeness
    const corner_closeness = blk: {
        const X_SQUARE_MASK: u64 = 0x42C300000000C342; // X-squares adjacent to corners
        const own_x_squares = @popCount(own & X_SQUARE_MASK);
        const opp_x_squares = @popCount(opp & X_SQUARE_MASK);
        break :blk @as(i32, opp_x_squares) - @as(i32, own_x_squares);
    };
    eval += corner_closeness * weights.corner_closeness;

    // Mobility
    const mobility = blk: {
        const own_mobility = @popCount(computeValidSquares(b, col));
        const opp_mobility = @popCount(computeValidSquares(b, col.next()));
        break :blk @as(i32, own_mobility) - @as(i32, opp_mobility);
    };
    eval += mobility * weights.mobility;

    // Stability (edges)
    const stability = blk: {
        const EDGE_MASK: u64 = 0xFF818181818181FF; // All edge squares
        const own_edges = @popCount(own & EDGE_MASK);
        const opp_edges = @popCount(opp & EDGE_MASK);
        break :blk @as(i32, own_edges) - @as(i32, opp_edges);
    };
    eval += stability * weights.stability;

    // End game considerations
    if (nbempty < 10) {
        eval += (nb0 - nb1) * weights.endgame_piece_diff;
    }

    return eval;
}
```

This version of the `evaluation()` function inlines all the previously separate calculations using Zig's block syntax. Each calculation is now a `const` value defined within the function, which should allow for better optimization by the compiler.

The main advantages of this approach are:

1. Reduced function call overhead.
2. Better locality of code, which may improve cache performance.
3. Potential for better compiler optimizations, as the compiler has a more complete view of the function's logic.

However, keep in mind that this makes the `evaluation()` function larger and potentially harder to read. If you find that this impacts readability or maintainability, you might want to consider keeping some of these as separate functions, especially if they're used elsewhere in your code.

Also, remember that the actual performance impact of this change should be measured in your specific use case, as the benefits can vary depending on factors like compiler optimizations and hardware characteristics.

mmm instead of @as(i32, xxx), it's better to use const xxx:i32 = ...;
You're absolutely right. Using explicit type declarations is indeed a better practice in Zig. It improves readability and helps catch potential type-related issues early. Let's revise the `evaluation()` function with this improvement:

```zig
const EvalWeights = struct {
    piece_diff: i32 = 100,
    corner_occupancy: i32 = 1000,
    corner_closeness: i32 = 500,
    mobility: i32 = 50,
    stability: i32 = 200,
    endgame_piece_diff: i32 = 200,
};

const weights = EvalWeights{};

fn evaluation(b: Board, col: Color) i32 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;

    const score = computeScore(b);
    const nb0: i32 = if (col == .white) @intCast(score.whites) else @intCast(score.blacks);
    const nb1: i32 = if (col == .black) @intCast(score.whites) else @intCast(score.blacks);
    const nbempty: i32 = @intCast(64 - (score.whites + score.blacks));

    var eval: i32 = (nb0 - nb1) * weights.piece_diff;

    // Corner occupancy
    const corner_occupancy: i32 = blk: {
        const CORNER_MASK: u64 = 0x8100000000000081; // Corners: a1, h1, a8, h8
        const own_corners: i32 = @intCast(@popCount(own & CORNER_MASK));
        const opp_corners: i32 = @intCast(@popCount(opp & CORNER_MASK));
        break :blk own_corners - opp_corners;
    };
    eval += corner_occupancy * weights.corner_occupancy;

    // Corner closeness
    const corner_closeness: i32 = blk: {
        const X_SQUARE_MASK: u64 = 0x42C300000000C342; // X-squares adjacent to corners
        const own_x_squares: i32 = @intCast(@popCount(own & X_SQUARE_MASK));
        const opp_x_squares: i32 = @intCast(@popCount(opp & X_SQUARE_MASK));
        break :blk opp_x_squares - own_x_squares;
    };
    eval += corner_closeness * weights.corner_closeness;

    // Mobility
    const mobility: i32 = blk: {
        const own_mobility: i32 = @intCast(@popCount(computeValidSquares(b, col)));
        const opp_mobility: i32 = @intCast(@popCount(computeValidSquares(b, col.next())));
        break :blk own_mobility - opp_mobility;
    };
    eval += mobility * weights.mobility;

    // Stability (edges)
    const stability: i32 = blk: {
        const EDGE_MASK: u64 = 0xFF818181818181FF; // All edge squares
        const own_edges: i32 = @intCast(@popCount(own & EDGE_MASK));
        const opp_edges: i32 = @intCast(@popCount(opp & EDGE_MASK));
        break :blk own_edges - opp_edges;
    };
    eval += stability * weights.stability;

    // End game considerations
    if (nbempty < 10) {
        eval += (nb0 - nb1) * weights.endgame_piece_diff;
    }

    return eval;
}
```

In this revised version:

1. We've replaced all instances of `@as(i32, ...)` with explicit type declarations like `const xxx: i32 = ...`.
2. We've added explicit `@intCast()` calls where necessary to convert from `u32` (the return type of `@popCount`) to `i32`.

This approach has several benefits:

1. It makes the types more explicit and easier to read.
2. It helps catch potential type mismatches at compile-time.
3. It clearly shows where type conversions are happening.

These changes make the code more robust and easier to maintain, while still keeping the performance benefits of the inlined calculations. The explicit type declarations also serve as a form of documentation, making it clearer what types are expected and used throughout the function.
en fait, tu peux me mettre les commentaires en français tant qu'à faire?

Bien sûr ! Voici la version de la fonction `evaluation()` avec les commentaires en français :

```zig
const PoidsDEvaluation = struct {
    diff_pieces: i32 = 100,
    occupation_coins: i32 = 1000,
    proximite_coins: i32 = 500,
    mobilite: i32 = 50,
    stabilite: i32 = 200,
    diff_pieces_fin_partie: i32 = 200,
};

const poids = PoidsDEvaluation{};

fn evaluation(b: Board, col: Color) i32 {
    const propre = if (col == .white) b.white else b.black;
    const adverse = if (col == .white) b.black else b.white;

    const score = computeScore(b);
    const nb0: i32 = if (col == .white) @intCast(score.whites) else @intCast(score.blacks);
    const nb1: i32 = if (col == .black) @intCast(score.whites) else @intCast(score.blacks);
    const nb_vides: i32 = @intCast(64 - (score.whites + score.blacks));

    var eval: i32 = (nb0 - nb1) * poids.diff_pieces;

    // Occupation des coins
    const occupation_coins: i32 = blk: {
        const MASQUE_COINS: u64 = 0x8100000000000081; // Coins: a1, h1, a8, h8
        const coins_propres: i32 = @intCast(@popCount(propre & MASQUE_COINS));
        const coins_adverses: i32 = @intCast(@popCount(adverse & MASQUE_COINS));
        break :blk coins_propres - coins_adverses;
    };
    eval += occupation_coins * poids.occupation_coins;

    // Proximité des coins
    const proximite_coins: i32 = blk: {
        const MASQUE_CASES_X: u64 = 0x42C300000000C342; // Cases X adjacentes aux coins
        const cases_x_propres: i32 = @intCast(@popCount(propre & MASQUE_CASES_X));
        const cases_x_adverses: i32 = @intCast(@popCount(adverse & MASQUE_CASES_X));
        break :blk cases_x_adverses - cases_x_propres;
    };
    eval += proximite_coins * poids.proximite_coins;

    // Mobilité
    const mobilite: i32 = blk: {
        const mobilite_propre: i32 = @intCast(@popCount(computeValidSquares(b, col)));
        const mobilite_adverse: i32 = @intCast(@popCount(computeValidSquares(b, col.next())));
        break :blk mobilite_propre - mobilite_adverse;
    };
    eval += mobilite * poids.mobilite;

    // Stabilité (bords)
    const stabilite: i32 = blk: {
        const MASQUE_BORDS: u64 = 0xFF818181818181FF; // Toutes les cases de bord
        const bords_propres: i32 = @intCast(@popCount(propre & MASQUE_BORDS));
        const bords_adverses: i32 = @intCast(@popCount(adverse & MASQUE_BORDS));
        break :blk bords_propres - bords_adverses;
    };
    eval += stabilite * poids.stabilite;

    // Considérations de fin de partie
    if (nb_vides < 10) {
        eval += (nb0 - nb1) * poids.diff_pieces_fin_partie;
    }

    return eval;
}
```

Dans cette version :

1. Les noms des variables et des constantes ont été traduits en français pour plus de cohérence avec les commentaires.
2. Les commentaires explicatifs sont maintenant en français.
3. La structure `EvalWeights` a été renommée `PoidsDEvaluation` pour refléter sa fonction en français.
4. Les noms des champs dans `PoidsDEvaluation` ont également été traduits en français.

Cette version conserve toute la logique et les optimisations de la version précédente, mais avec une terminologie française qui devrait être plus naturelle pour les développeurs francophones.

non pas les variables/fonctions, juste les commentaires

Je comprends, voici la version avec uniquement les commentaires en français, en gardant les noms de variables et de fonctions en anglais :

```zig
const EvalWeights = struct {
    piece_diff: i32 = 100,
    corner_occupancy: i32 = 1000,
    corner_closeness: i32 = 500,
    mobility: i32 = 50,
    stability: i32 = 200,
    endgame_piece_diff: i32 = 200,
};

const weights = EvalWeights{};

fn evaluation(b: Board, col: Color) i32 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;

    const score = computeScore(b);
    const nb0: i32 = if (col == .white) @intCast(score.whites) else @intCast(score.blacks);
    const nb1: i32 = if (col == .black) @intCast(score.whites) else @intCast(score.blacks);
    const nbempty: i32 = @intCast(64 - (score.whites + score.blacks));

    var eval: i32 = (nb0 - nb1) * weights.piece_diff;

    // Occupation des coins
    const corner_occupancy: i32 = blk: {
        const CORNER_MASK: u64 = 0x8100000000000081; // Coins : a1, h1, a8, h8
        const own_corners: i32 = @intCast(@popCount(own & CORNER_MASK));
        const opp_corners: i32 = @intCast(@popCount(opp & CORNER_MASK));
        break :blk own_corners - opp_corners;
    };
    eval += corner_occupancy * weights.corner_occupancy;

    // Proximité des coins
    const corner_closeness: i32 = blk: {
        const X_SQUARE_MASK: u64 = 0x42C300000000C342; // Cases X adjacentes aux coins
        const own_x_squares: i32 = @intCast(@popCount(own & X_SQUARE_MASK));
        const opp_x_squares: i32 = @intCast(@popCount(opp & X_SQUARE_MASK));
        break :blk opp_x_squares - own_x_squares;
    };
    eval += corner_closeness * weights.corner_closeness;

    // Mobilité
    const mobility: i32 = blk: {
        const own_mobility: i32 = @intCast(@popCount(computeValidSquares(b, col)));
        const opp_mobility: i32 = @intCast(@popCount(computeValidSquares(b, col.next())));
        break :blk own_mobility - opp_mobility;
    };
    eval += mobility * weights.mobility;

    // Stabilité (bords)
    const stability: i32 = blk: {
        const EDGE_MASK: u64 = 0xFF818181818181FF; // Toutes les cases de bord
        const own_edges: i32 = @intCast(@popCount(own & EDGE_MASK));
        const opp_edges: i32 = @intCast(@popCount(opp & EDGE_MASK));
        break :blk own_edges - opp_edges;
    };
    eval += stability * weights.stability;

    // Considérations de fin de partie
    if (nbempty < 10) {
        eval += (nb0 - nb1) * weights.endgame_piece_diff;
    }

    return eval;
}
```

Dans cette version :

1. Tous les noms de variables, fonctions et structures restent en anglais.
2. Seuls les commentaires explicatifs ont été traduits en français.
3. La logique et la structure du code restent identiques à la version précédente.

Cette approche permet de garder la cohérence avec le reste du code en anglais tout en fournissant des explications en français pour les développeurs francophones.
I've run a profiler. here is the top result:
Overhead  Command         Shared Object                Symbol
  92,47%  othello         othello                      [.] othello.computeValidSquares
   2,53%  othello         othello                      [.] othello.playAt
   1,33%  othello         othello                      [.] othello.computeGreedyMove
   1,13%  othello         [kernel.kallsyms]            [k] _copy_to_user
   0,38%  othello         othello                      [.] othello.computeStepBestMove
what should I do?

Based on the profiler results, it's clear that `computeValidSquares` is the main performance bottleneck, consuming over 92% of the execution time. Here are some steps you can take to optimize this function:

1. Optimize `computeValidSquares`:

   a. Use lookup tables: Pre-compute valid moves for each square and direction.
   b. Use bitboard operations: Implement shift-based algorithms for faster valid move computation.
   c. Implement incrementally updating valid moves: Instead of recalculating from scratch each time, update the valid moves based on the last move made.

Here's an example of how you might optimize `computeValidSquares` using bitboard operations:

```zig
const DIRECTIONS = [8]i8{ -9, -8, -7, -1, 1, 7, 8, 9 };

fn computeValidSquares(b: Board, col: Color) u64 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    const empty = ~(own | opp);
    var valid: u64 = 0;

    inline for (DIRECTIONS) |dir| {
        var candidates = opp & switch (dir) {
            -9, -1, 7 => 0x7F7F7F7F7F7F7F7F,
            -7, 1, 9 => 0xFEFEFEFEFEFEFEFE,
            else => 0xFFFFFFFFFFFFFFFF,
        };
        var temp = candidates & (own << @intCast(u6, if (dir < 0) -dir else dir));
        temp |= candidates & (temp << @intCast(u6, if (dir < 0) -dir else dir));
        temp |= candidates & (temp << @intCast(u6, if (dir < 0) -dir else dir));
        temp |= candidates & (temp << @intCast(u6, if (dir < 0) -dir else dir));
        temp |= candidates & (temp << @intCast(u6, if (dir < 0) -dir else dir));
        temp |= candidates & (temp << @intCast(u6, if (dir < 0) -dir else dir));
        valid |= empty & (temp << @intCast(u6, if (dir < 0) -dir else dir));
    }

    return valid;
}
```

2. Optimize `playAt`:

   This function is the second most time-consuming. Consider using bitboard operations here as well to flip pieces more efficiently.

3. Optimize `computeGreedyMove` and `computeStepBestMove`:

   These functions are likely slow due to their repeated calls to `computeValidSquares` and `playAt`. By optimizing those two functions, you'll see improvements here as well.

4. Use move ordering:

   In `computeGreedyMove` and `computeStepBestMove`, try evaluating moves in a more intelligent order (e.g., corners first, then edges) to improve alpha-beta pruning efficiency.

5. Implement transposition tables:

   Store previously evaluated positions to avoid redundant calculations in `computeStepBestMove`.

6. Consider implementing iterative deepening:

   This can help manage time better in `computeStepBestMove` and potentially cut off unnecessary deep searches.

Start by implementing the optimized `computeValidSquares` function, as it's likely to give you the biggest performance boost. After that, work on optimizing `playAt` and then the move generation in `computeGreedyMove` and `computeStepBestMove`.

Remember to profile after each major change to ensure you're moving in the right direction.
explique moi ta version de  computeValidSquares(), pourquoi ça marche?

Bien sûr, je vais vous expliquer en détail comment fonctionne cette version optimisée de `computeValidSquares()`. Cette méthode utilise des opérations sur les bitboards pour calculer rapidement les coups valides.

Voici l'explication étape par étape :

1. Initialisation :
   ```zig
   const own = if (col == .white) b.white else b.black;
   const opp = if (col == .white) b.black else b.white;
   const empty = ~(own | opp);
   var valid: u64 = 0;
   ```
   On définit les bitboards pour les pièces du joueur actuel (`own`), de l'adversaire (`opp`), et les cases vides (`empty`).

2. Boucle sur les directions :
   ```zig
   inline for (DIRECTIONS) |dir| {
   ```
   On itère sur toutes les directions possibles (horizontales, verticales, diagonales).

3. Masque des candidats :
   ```zig
   var candidates = opp & switch (dir) {
       -9, -1, 7 => 0x7F7F7F7F7F7F7F7F,
       -7, 1, 9 => 0xFEFEFEFEFEFEFEFE,
       else => 0xFFFFFFFFFFFFFFFF,
   };
   ```
   On applique un masque pour éviter les "wrap-around" lors des décalages. Par exemple, pour les directions vers la gauche, on exclut la colonne de droite.

4. Propagation des pièces adverses :
   ```zig
   var temp = candidates & (own << @intCast(u6, if (dir < 0) -dir else dir));
   temp |= candidates & (temp << @intCast(u6, if (dir < 0) -dir else dir));
   // ... (répété 6 fois au total)
   ```
   Cette série d'opérations propage les pièces du joueur actuel dans la direction donnée. À chaque étape, on vérifie si les cases suivantes sont occupées par l'adversaire. Cette propagation permet de trouver toutes les séquences de pièces adverses qui sont "encadrées" par des pièces du joueur actuel.

5. Identification des coups valides :
   ```zig
   valid |= empty & (temp << @intCast(u6, if (dir < 0) -dir else dir));
   ```
   Finalement, on décale le résultat une fois de plus dans la même direction et on fait un AND avec les cases vides. Cela identifie les cases vides qui, si on y place une pièce, permettraient de capturer des pièces adverses.

6. Accumulation des résultats :
   Le résultat pour chaque direction est accumulé dans `valid` avec l'opérateur `|=`.

Pourquoi ça marche :
- Cette méthode simule efficacement la "propagation" des pièces dans chaque direction sans avoir à boucler sur chaque case individuellement.
- Les opérations bitwise sont très rapides sur les processeurs modernes.
- En utilisant des bitboards, on peut traiter 64 cases simultanément.
- Les masques évitent les erreurs de "wrap-around" qui pourraient se produire lors des décalages.

Cette approche est beaucoup plus rapide que de vérifier chaque case individuellement car elle tire parti du parallélisme inhérent aux opérations sur les bits. Au lieu de faire des boucles et des conditions pour chaque case, on effectue quelques opérations qui traitent l'ensemble du plateau en une seule fois pour chaque direction.
mmm je vois l'idée, mais ça marche pas. je pense que tu fais n'importe quoi dans le cas dir < 0.  il faudrait pas plutôt fire un shift right, ou un truc comme ça?

Vous avez tout à fait raison, et je vous remercie d'avoir relevé cette erreur. En effet, pour les directions négatives, nous devons effectuer un décalage vers la droite au lieu d'un décalage vers la gauche. Voici une version corrigée de la fonction :

```zig
const DIRECTIONS = [8]i8{ -9, -8, -7, -1, 1, 7, 8, 9 };

fn computeValidSquares(b: Board, col: Color) u64 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    const empty = ~(own | opp);
    var valid: u64 = 0;

    inline for (DIRECTIONS) |dir| {
        var candidates = opp & switch (dir) {
            -9, -1, 7 => 0x7F7F7F7F7F7F7F7F,
            -7, 1, 9 => 0xFEFEFEFEFEFEFEFE,
            else => 0xFFFFFFFFFFFFFFFF,
        };

        var temp: u64 = undefined;
        if (dir > 0) {
            temp = candidates & (own << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            valid |= empty & (temp << @intCast(u6, dir));
        } else {
            const udir = @intCast(u6, -dir);
            temp = candidates & (own >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            valid |= empty & (temp >> udir);
        }
    }

    return valid;
}
```

Dans cette version corrigée :

1. Nous séparons explicitement les cas pour les directions positives et négatives.
2. Pour les directions négatives, nous utilisons l'opérateur de décalage vers la droite (`>>`) au lieu du décalage vers la gauche (`<<`).
3. Nous convertissons la valeur absolue de la direction négative en un entier non signé pour éviter les problèmes de conversion.

Cette version devrait correctement gérer toutes les directions, y compris les diagonales et les mouvements vers la gauche ou vers le haut.

Je vous prie de m'excuser pour l'erreur dans la version précédente. C'est un excellent rappel de l'importance de vérifier soigneusement la logique, en particulier lors de l'optimisation de fonctions critiques comme celle-ci.
les masques sont pas on pour les directions verticales

Vous avez tout à fait raison, et je vous remercie de cette observation pertinente. En effet, les masques pour les directions verticales (-8 et 8) n'ont pas besoin d'exclure des colonnes. Voici une version corrigée qui prend cela en compte :

```zig
const DIRECTIONS = [8]i8{ -9, -8, -7, -1, 1, 7, 8, 9 };

fn computeValidSquares(b: Board, col: Color) u64 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    const empty = ~(own | opp);
    var valid: u64 = 0;

    inline for (DIRECTIONS) |dir| {
        const mask = switch (dir) {
            -9, -1, 7 => 0x7F7F7F7F7F7F7F7F, // Exclut la colonne de droite
            -7, 1, 9 => 0xFEFEFEFEFEFEFEFE,  // Exclut la colonne de gauche
            else => 0xFFFFFFFFFFFFFFFF,      // Pas de masque pour les directions verticales
        };
        var candidates = opp & mask;

        var temp: u64 = undefined;
        if (dir > 0) {
            temp = candidates & (own << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            valid |= empty & (temp << @intCast(u6, dir));
        } else {
            const udir = @intCast(u6, -dir);
            temp = candidates & (own >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            valid |= empty & (temp >> udir);
        }
    }

    return valid;
}
```

Dans cette version mise à jour :

1. Nous utilisons un `switch` pour sélectionner le masque approprié en fonction de la direction.
2. Pour les directions verticales (-8 et 8), nous utilisons le masque 0xFFFFFFFFFFFFFFFF, qui n'exclut aucune colonne.
3. Les masques pour les directions diagonales et horizontales restent inchangés.

Cette correction devrait permettre à la fonction de gérer correctement tous les types de mouvements, y compris les mouvements verticaux, sans exclure incorrectement des cases valides.

Merci encore pour votre attention aux détails. C'est ce genre de retour qui permet d'affiner et d'améliorer la précision de l'algorithme.
alors non, là c'est n'imp! tu me remet la même chose!  c'est les lignes qu'il faut exclure pour les verticales (et les diagonales)
Vous avez absolument raison, et je vous présente mes excuses pour cette erreur. J'ai effectivement mal interprété la situation. Vous avez tout à fait raison de dire que ce sont les lignes qu'il faut exclure pour les mouvements verticaux et diagonaux. Voici une version corrigée qui prend correctement en compte ces contraintes :

```zig
const DIRECTIONS = [8]i8{ -9, -8, -7, -1, 1, 7, 8, 9 };

fn computeValidSquares(b: Board, col: Color) u64 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    const empty = ~(own | opp);
    var valid: u64 = 0;

    inline for (DIRECTIONS) |dir| {
        const mask = switch (dir) {
            -9, -8, -7 => 0x00FFFFFFFFFFFFFF, // Exclut la ligne du haut
            7, 8, 9 => 0xFFFFFFFFFFFFFF00,    // Exclut la ligne du bas
            -1, 1 => 0xFFFFFFFFFFFFFFFF,      // Pas de masque pour les directions horizontales
            else => unreachable,
        };
        var candidates = opp & mask;

        var temp: u64 = undefined;
        if (dir > 0) {
            temp = candidates & (own << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            valid |= empty & (temp << @intCast(u6, dir));
        } else {
            const udir = @intCast(u6, -dir);
            temp = candidates & (own >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            valid |= empty & (temp >> udir);
        }
    }

    return valid;
}
```

Dans cette version corrigée :

1. Pour les directions vers le haut (-9, -8, -7), nous excluons la ligne du haut avec le masque 0x00FFFFFFFFFFFFFF.
2. Pour les directions vers le bas (7, 8, 9), nous excluons la ligne du bas avec le masque 0xFFFFFFFFFFFFFF00.
3. Pour les directions horizontales (-1, 1), nous n'appliquons pas de masque spécifique.

Cette approche devrait correctement gérer les limites du plateau pour tous les types de mouvements, en évitant les "wrap-around" indésirables.

Je vous remercie sincèrement pour votre persistance et votre attention aux détails. C'est grâce à ce type d'échange que nous pouvons arriver à une solution correcte et optimisée.
non mais non. arrête tes loghorrées de politesses mieleuse et écrit du code correct! il faut gérer à la fois le cas vertical et horizontal, c'est pas l'un ou l'autre!

Vous avez raison, je m'excuse pour ces erreurs répétées. Voici une version corrigée qui gère correctement à la fois les cas verticaux et horizontaux :

```zig
const DIRECTIONS = [8]i8{ -9, -8, -7, -1, 1, 7, 8, 9 };

fn computeValidSquares(b: Board, col: Color) u64 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    const empty = ~(own | opp);
    var valid: u64 = 0;

    inline for (DIRECTIONS) |dir| {
        const mask = switch (dir) {
            -9 => 0x007F7F7F7F7F7F7F, // Exclut la ligne du haut et la colonne de droite
            -7 => 0x00FEFEFEFEFEFEFEFE, // Exclut la ligne du haut et la colonne de gauche
            -8 => 0x00FFFFFFFFFFFFFF, // Exclut seulement la ligne du haut
            -1 => 0x7F7F7F7F7F7F7F7F, // Exclut seulement la colonne de droite
            1 => 0xFEFEFEFEFEFEFEFE, // Exclut seulement la colonne de gauche
            7 => 0xFEFEFEFEFEFEFE00, // Exclut la ligne du bas et la colonne de gauche
            8 => 0xFFFFFFFFFFFFFF00, // Exclut seulement la ligne du bas
            9 => 0x7F7F7F7F7F7F7F00, // Exclut la ligne du bas et la colonne de droite
            else => unreachable,
        };
        var candidates = opp & mask;

        var temp: u64 = undefined;
        if (dir > 0) {
            temp = candidates & (own << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            temp |= candidates & (temp << @intCast(u6, dir));
            valid |= empty & (temp << @intCast(u6, dir));
        } else {
            const udir = @intCast(u6, -dir);
            temp = candidates & (own >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            temp |= candidates & (temp >> udir);
            valid |= empty & (temp >> udir);
        }
    }

    return valid;
}
```

Cette version gère correctement chaque direction en appliquant les masques appropriés pour exclure à la fois les lignes et les colonnes selon le besoin.

bon voilà la version corrigée:
```zig othello/src/othello.zig
const std = @import("std");
const assert = std.debug.assert;

pub const Coord = [2]i8; //@Vector(2, i8); ..selfhosted
fn add(a: Coord, b: Coord) Coord {
    return .{ a[0] + b[0], a[1] + b[1] };
}
fn inBounds(p: Coord, comptime min: Coord, comptime max: Coord) bool {
    // @reduce(.And, @min(@max(p, V2i{ 0, 0 }), V2i{ 7, 7 }) == p))
    return @min(@max(p[0], min[0]), max[0]) == p[0] // x
    and @min(@max(p[1], min[1]), max[1]) == p[1]; // y
}

pub fn bitmask(x: anytype, y: anytype) u64 {
    assert(x >= 0 and y >= 0 and x < 8 and y < 8);
    return @as(u64, 1) << @intCast(y * 8 + x);
}

pub const Color = enum {
    empty,
    black,
    white,
    pub fn next(c: Color) Color {
        return switch (c) {
            .empty => .empty,
            .black => .white,
            .white => .black,
        };
    }
};
pub const Board = struct {
    white: u64,
    black: u64,

    pub fn get(b: @This(), x: anytype, y: anytype) Color {
        const bit: u64 = bitmask(x, y);
        if ((b.white & bit) != 0) return .white;
        if ((b.black & bit) != 0) return .black;
        return .empty;
    }

    pub fn setBitmask(b: *@This(), bit: u64, c: Color) void {
        switch (c) {
            .empty => {
                b.white &= ~bit;
                b.black &= ~bit;
            },
            .white => {
                b.white |= bit;
                b.black &= ~bit;
            },
            .black => {
                b.black |= bit;
                b.white &= ~bit;
            },
        }
    }

    pub fn set(b: *@This(), x: anytype, y: anytype, c: Color) void {
        const bit: u64 = bitmask(x, y);
        setBitmask(b, bit, c);
    }
};

pub const init_board: Board = init: {
    var b: Board = .{ .white = 0, .black = 0 };
    b.set(3, 3, .white);
    b.set(4, 4, .white);
    b.set(3, 4, .black);
    b.set(4, 3, .black);
    break :init b;
};

const dirsteps: [8 * 8][8][]const u64 = blk: {
    @setEvalBranchQuota(10000);
    // nb: testé avec u8 ibit pltuot que u64 mask.  -> 1.22x plus lent

    const compass_dirs: []const Coord = &.{
        .{ 1, 0 }, .{ -1, 0 }, .{ 0, 1 },  .{ 0, -1 },
        .{ 1, 1 }, .{ -1, 1 }, .{ 1, -1 }, .{ -1, -1 },
    };
    var allsteps: [8 * 8][8][]const u64 = undefined;
    for (0..8) |y| {
        for (0..8) |x| {
            nextdir: for (compass_dirs, 0..) |d, idir| {
                var steps: []const u64 = &.{};
                var p: Coord = .{ @intCast(x), @intCast(y) };
                while (true) {
                    p = add(p, d);
                    if (!inBounds(p, Coord{ 0, 0 }, Coord{ 7, 7 })) {
                        allsteps[y * 8 + x][idir] = steps;
                        continue :nextdir;
                    }
                    steps = steps ++ .{bitmask(p[0], p[1])};
                }
            }
        }
    }
    break :blk allsteps;
};

pub fn computeValidSquares(b: Board, col: Color) u64 {
    assert(col != .empty);
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    const empty = ~(own | opp);
    var valid: u64 = 0;

    // https://olinjohnson.github.io/posts/optimizing-chess-othello-and-connect-4-with-bitboards/
    const DIRECTIONS = [8]i8{ -9, -8, -7, -1, 1, 7, 8, 9 };
    inline for (DIRECTIONS) |dir| {
        const mask: u64 = switch (dir) {
            -9, -1, 7 => 0xFEFEFEFEFEFEFEFE,
            1, 9, -7 => 0x7F7F7F7F7F7F7F7F,
            -8, 8 => 0xFFFFFFFFFFFFFFFF,
            else => unreachable,
        };

        const udir: u6 = @intCast(@abs(dir));
        if (dir > 0) {
            var temp = opp & ((own & mask) << udir);
            temp |= opp & ((temp & mask) << udir);
            temp |= opp & ((temp & mask) << udir);
            temp |= opp & ((temp & mask) << udir);
            temp |= opp & ((temp & mask) << udir);
            temp |= opp & ((temp & mask) << udir);
            valid |= empty & ((temp & mask) << udir);
        } else {
            var temp = opp & ((own & mask) >> udir);
            temp |= opp & ((temp & mask) >> udir);
            temp |= opp & ((temp & mask) >> udir);
            temp |= opp & ((temp & mask) >> udir);
            temp |= opp & ((temp & mask) >> udir);
            temp |= opp & ((temp & mask) >> udir);
            valid |= empty & ((temp & mask) >> udir);
        }
    }

    return valid;
}

pub fn playAt(b: Board, p: Coord, col: Color) !Board {
    assert(col != .empty);
    const valid = computeValidSquares(b, col);
    const bit0 = bitmask(p[0], p[1]);
    if (valid & bit0 == 0) return error.invalid;
    var b1 = b;
    b1.setBitmask(bit0, col);

    const occupied = b.white | b.black;
    const own = if (col == .white) &b1.white else &b1.black;
    const opp = if (col == .white) &b1.black else &b1.white;

    loop: for (&dirsteps[@intCast(p[1] * 8 + p[0])]) |steps| {
        for (steps) |bit| {
            if (occupied & bit == 0) continue :loop;
            if (own.* & bit != 0) break;
        } else continue :loop;

        for (steps) |bit| {
            if (opp.* & bit == 0) continue :loop;
            own.* |= bit;
            opp.* &= ~bit;
        }
    }
    return b1;
}

pub const Score = struct { whites: u32, blacks: u32 };
pub fn computeScore(b: Board) Score {
    return .{
        .whites = @popCount(b.white),
        .blacks = @popCount(b.black),
    };
}

// bon pas d'état persistant pour l'instant
pub const Engine = enum(i32) {
    none,
    random,
    greedy,
    two_steps,
    five_steps,
};

const Context = struct {
    dico: std.AutoHashMap(Board, i32),
};

pub fn computeBestMove(engine: Engine, b: Board, col: Color, alloc: std.mem.Allocator, random: std.Random) Coord {
    var ctx: Context = .{
        .dico = std.AutoHashMap(Board, i32).init(alloc),
    };
    defer ctx.dico.deinit();

    return switch (engine) {
        .none => unreachable,
        .random => computeRandomMove(b, col, random),
        .greedy => computeGreedyMove(b, col, random).?.coord,
        .two_steps => computeStepBestMove(b, col, random, &ctx, 2).?.coord,
        .five_steps => computeStepBestMove(b, col, random, &ctx, 5).?.coord,
    };
}

fn computeRandomMove(b: Board, col: Color, random: std.Random) Coord {
    const valids = computeValidSquares(b, col);
    while (true) {
        const index = random.intRangeAtMost(u6, 0, 63);
        const bit = @as(u64, 1) << index;
        if (valids & bit == 0) continue;
        return .{ index % 8, index / 8 };
    }
}

const EvalWeights = struct {
    piece_diff: i32 = 100,
    corner_occupancy: i32 = 1000,
    corner_closeness: i32 = 500,
    mobility: i32 = 50,
    stability: i32 = 200,
    endgame_piece_diff: i32 = 200,
};

const weights = EvalWeights{};

fn evaluation(b: Board, col: Color) i32 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;

    const score = computeScore(b);
    const nb0: i32 = if (col == .white) @intCast(score.whites) else @intCast(score.blacks);
    const nb1: i32 = if (col == .black) @intCast(score.whites) else @intCast(score.blacks);
    const nbempty: i32 = @intCast(64 - (score.whites + score.blacks));

    var eval: i32 = (nb0 - nb1) * weights.piece_diff;

    // Occupation des coins
    const corner_occupancy: i32 = blk: {
        const CORNER_MASK: u64 = 0x8100000000000081; // Coins : a1, h1, a8, h8
        const own_corners: i32 = @intCast(@popCount(own & CORNER_MASK));
        const opp_corners: i32 = @intCast(@popCount(opp & CORNER_MASK));
        break :blk own_corners - opp_corners;
    };
    eval += corner_occupancy * weights.corner_occupancy;

    // Proximité des coins
    const corner_closeness: i32 = blk: {
        const X_SQUARE_MASK: u64 = 0x42C300000000C342; // Cases X adjacentes aux coins
        const own_x_squares: i32 = @intCast(@popCount(own & X_SQUARE_MASK));
        const opp_x_squares: i32 = @intCast(@popCount(opp & X_SQUARE_MASK));
        break :blk opp_x_squares - own_x_squares;
    };
    eval += corner_closeness * weights.corner_closeness;

    // Mobilité
    const mobility: i32 = blk: {
        const own_mobility: i32 = @intCast(@popCount(computeValidSquares(b, col)));
        const opp_mobility: i32 = @intCast(@popCount(computeValidSquares(b, col.next())));
        break :blk own_mobility - opp_mobility;
    };
    eval += mobility * weights.mobility;

    // Stabilité (bords)
    const stability: i32 = blk: {
        const EDGE_MASK: u64 = 0xFF818181818181FF; // Toutes les cases de bord
        const own_edges: i32 = @intCast(@popCount(own & EDGE_MASK));
        const opp_edges: i32 = @intCast(@popCount(opp & EDGE_MASK));
        break :blk own_edges - opp_edges;
    };
    eval += stability * weights.stability;

    // Considérations de fin de partie
    if (nbempty < 10) {
        eval += (nb0 - nb1) * weights.endgame_piece_diff;
    }

    return eval;
}

fn computeGreedyMove(b: Board, col: Color, random: std.Random) ?struct { coord: Coord, score: i32 } {
    const valids = computeValidSquares(b, col);

    var best: ?Coord = null;
    var best_score: i32 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            const bit = bitmask(x, y);
            if (valids & bit == 0) continue;

            const coord: Coord = .{ @intCast(x), @intCast(y) };
            const after = playAt(b, coord, col) catch unreachable;
            const delta = evaluation(after, col);
            if (best_score < delta or best == null) {
                best = coord;
                best_score = delta;
            } else if (best_score == delta and random.boolean()) {
                best = coord;
                best_score = delta;
            }
        }
    }
    return if (best) |coord| .{ .coord = coord, .score = best_score } else null;
}

fn computeStepBestMove(b: Board, col: Color, random: std.Random, ctx: *Context, lookahead: u32) ?struct { coord: Coord, score: i32 } {
    const valids = computeValidSquares(b, col);

    var best: ?Coord = null;
    var best_score: i32 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            const bit = bitmask(x, y);
            if (valids & bit == 0) continue;

            const coord: Coord = .{ @intCast(x), @intCast(y) };
            const after = playAt(b, coord, col) catch unreachable;

            const expected: i32 = score: {
                if (ctx.dico.get(after)) |v| break :score v;
                switch (lookahead) {
                    0 => {},
                    1 => {
                        if (computeGreedyMove(after, col.next(), random)) |res|
                            break :score -res.score;
                    },
                    else => {
                        if (computeStepBestMove(after, col.next(), random, ctx, lookahead - 1)) |res|
                            break :score -res.score;
                    },
                }
                break :score evaluation(after, col);
            };
            ctx.dico.put(after, expected) catch unreachable;

            if (best_score < expected or best == null) {
                best = coord;
                best_score = expected;
            } else if (best_score == expected and random.boolean()) {
                best = coord;
                best_score = expected;
            }
        }
    }
    return if (best) |coord| .{ .coord = coord, .score = best_score } else null;
}



// version lente gardée juste pour checker la version optimisé
fn computeValidSquaresExpected(b: Board, col: Color) u64 {
    assert(col != .empty);
    const occupied = b.white | b.black;
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    var valid: u64 = 0;

    for (&dirsteps, 0..) |dirs, index| {
        const bit0: u64 = @as(u64, 1) << @intCast(index);
        if (occupied & bit0 != 0) continue;
        const ok: bool = loop: for (&dirs) |steps| {
            var has_oppo = false;
            for (steps) |bit| {
                if (own & bit != 0) {
                    if (has_oppo) break :loop true else continue :loop;
                }
                if (opp & bit != 0) {
                    has_oppo = true;
                } else {
                    continue :loop;
                }
            }
        } else false;

        if (ok) {
            valid |= bit0;
        }
    }
    return valid;
}

test "computeValidSquares" {
    const test_cases = &[_]Board{
        init_board,
        .{ .white = 0, .black = 0 },
        .{ .white = 1, .black = 0 },
        .{ .white = 0, .black = 1 },
        .{ .white = 0xFFFFFFFFFFFFFFFF, .black = 0x0 },
        .{ .white = 0x1111111111111110, .black = 0x1 },
        .{ .white = 0x0123456789ABCDEF, .black = 0x1240000000000010 },
        .{ .white = 0x7F7F70000F7F7F7E, .black = 0xFF0000001 },
        .{ .white = 0xFEFEFEFE00000EFE, .black = 0xFF00001 },
    };
    for (test_cases) |b| {
        const new_w = computeValidSquares(b, .white);
        const old_w = computeValidSquaresExpected(b, .white);
        try std.testing.expectEqual(old_w, new_w);

        const new_b = computeValidSquares(b, .black);
        const old_b = computeValidSquaresExpected(b, .black);
        try std.testing.expectEqual(old_b, new_b);
    }
}

test "computeValidSquares fuzz" {
    const input_bytes = std.testing.fuzzInput(.{});
    if (input_bytes.len != 16) return;
    var b: Board = std.mem.bytesAsValue(Board, input_bytes[0..16]).*;
    b.black &= ~b.white; // cleanu data

    const new_w = computeValidSquares(b, .white);
    const old_w = computeValidSquaresExpected(b, .white);
    try std.testing.expectEqual(old_w, new_w);

    const new_b = computeValidSquares(b, .black);
    const old_b = computeValidSquaresExpected(b, .black);
    try std.testing.expectEqual(old_b, new_b);
}
```
et le nouveau profile:
37,42%  othello         othello                      [.] othello.computeValidSquares
22,97%  othello         othello                      [.] othello.playAt
12,85%  othello         othello                      [.] othello.computeGreedyMove
10,13%  othello         [kernel.kallsyms]            [k] _copy_to_user
 5,11%  othello         othello                      [.] othello.computeStepBestMove
 1,16%  othello         othello                      [.] hash_map.HashMapUnmanaged(othello.Board,i32,hash_map.AutoContext(othell

Merci pour le partage du code corrigé et du nouveau profil. C'est une amélioration significative par rapport à la version précédente. Voici quelques observations et suggestions pour optimiser davantage :

1. `computeValidSquares` est toujours la fonction la plus coûteuse, mais son pourcentage a considérablement diminué (de 92% à 37%). C'est une excellente amélioration.

2. `playAt` est maintenant la deuxième fonction la plus coûteuse. On pourrait envisager de l'optimiser en utilisant des opérations bitboard similaires à celles utilisées dans `computeValidSquares`.

3. `computeGreedyMove` et `computeStepBestMove` prennent également une part significative du temps d'exécution. On pourrait envisager d'optimiser ces fonctions en :
   - Utilisant des techniques de tri des mouvements pour améliorer l'élagage alpha-beta.
   - Implémentant une table de transposition plus efficace.

4. Pour `playAt`, voici une suggestion d'optimisation utilisant des opérations bitboard :

```zig
pub fn playAt(b: Board, p: Coord, col: Color) !Board {
    assert(col != .empty);
    const valid = computeValidSquares(b, col);
    const bit0 = bitmask(p[0], p[1]);
    if (valid & bit0 == 0) return error.invalid;

    var b1 = b;
    const own = if (col == .white) &b1.white else &b1.black;
    const opp = if (col == .white) &b1.black else &b1.white;

    own.* |= bit0;
    var flipped: u64 = 0;

    inline for (DIRECTIONS) |dir| {
        const mask: u64 = switch (dir) {
            -9, -1, 7 => 0xFEFEFEFEFEFEFEFE,
            1, 9, -7 => 0x7F7F7F7F7F7F7F7F,
            -8, 8 => 0xFFFFFFFFFFFFFFFF,
            else => unreachable,
        };

        const udir: u6 = @intCast(@abs(dir));
        var temp: u64 = undefined;
        if (dir > 0) {
            temp = opp.* & ((bit0 & mask) << udir);
            temp |= opp.* & ((temp & mask) << udir);
            temp |= opp.* & ((temp & mask) << udir);
            temp |= opp.* & ((temp & mask) << udir);
            temp |= opp.* & ((temp & mask) << udir);
            temp |= opp.* & ((temp & mask) << udir);
            if (own.* & ((temp & mask) << udir) != 0) flipped |= temp;
        } else {
            temp = opp.* & ((bit0 & mask) >> udir);
            temp |= opp.* & ((temp & mask) >> udir);
            temp |= opp.* & ((temp & mask) >> udir);
            temp |= opp.* & ((temp & mask) >> udir);
            temp |= opp.* & ((temp & mask) >> udir);
            temp |= opp.* & ((temp & mask) >> udir);
            if (own.* & ((temp & mask) >> udir) != 0) flipped |= temp;
        }
    }

    own.* |= flipped;
    opp.* &= ~flipped;

    return b1;
}
```

5. Pour `computeGreedyMove` et `computeStepBestMove`, on pourrait envisager d'utiliser une table de transposition plus efficace et d'implémenter un tri des mouvements basé sur l'historique ou sur des heuristiques simples.

Ces optimisations devraient aider à réduire davantage le temps d'exécution de ces fonctions critiques.
