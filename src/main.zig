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

    const main_board_pos: Vec2 = .{ .x = 20, .y = 20 };
    const main_board_size: f32 = 640;

    var board = othello.init_board;
    var nextcol: othello.Color = .black;
    var showhelpers = true;
    var game_over = false;
    var ai_engine = [_]othello.Engine{ undefined, .greedy, .none };

    var gui_dropdowns_states = [_]bool{ false, false };

    // Main game loop
    while (!rl.windowShouldClose()) { // Detect window close button or ESC key
        _ = frame_arena.reset(.retain_capacity);
        const frame_alloc = frame_arena.allocator();

        if (!game_over) {
            const can_play = othello.computeValidSquares(board, nextcol) != 0;
            if (!can_play) {
                nextcol = if (nextcol == .white) .black else .white;
                if (othello.computeValidSquares(board, nextcol) == 0)
                    game_over = true;
            }
        }

        const clicked_square: ?othello.Coord = sq: {
            if (game_over) break :sq null;
            const engine = ai_engine[@intFromEnum(nextcol)];
            if (engine == .none) {
                if (rl.isMouseButtonPressed(rl.MouseButton.mouse_button_left)) {
                    const p = Vec2.multiply(
                        Vec2.subtract(rl.getMousePosition(), main_board_pos),
                        .{ .x = 8.0 / main_board_size, .y = 8.0 / main_board_size },
                    );
                    if (p.x >= 0 and p.x < 8 and p.y >= 0 and p.y < 8)
                        break :sq .{ @intFromFloat(p.x), @intFromFloat(p.y) };
                }
                break :sq null;
            } else {
                break :sq othello.computeBestMove(engine, board, nextcol, frame_alloc, random);
            }
        };

        if (clicked_square) |sq| play: {
            board = othello.playAt(board, sq, nextcol) catch break :play;
            nextcol = if (nextcol == .white) .black else .white;
        }

        // Draw
        //----------------------------------------------------------------------------------
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.init(0, 33, 66, 255));
        drawBoard(board, main_board_pos, main_board_size);
        if (showhelpers)
            drawHelper(board, nextcol, main_board_pos, main_board_size);

        rl.drawText("Next: ", 700, 50, 30, rl.Color.light_gray);
        if (!game_over)
            drawPawn(.{ .x = 815, .y = 65 }, 20, nextcol);

        const score = othello.computeScore(board);
        const scoretxt = try std.fmt.allocPrintZ(frame_alloc, "Score: {} - {}", .{ score.blacks, score.whites });
        rl.drawText(scoretxt, 700, 100, 30, rl.Color.light_gray);

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
                board = othello.init_board;
                nextcol = .black;
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
