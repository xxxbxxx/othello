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
