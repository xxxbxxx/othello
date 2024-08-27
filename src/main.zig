// raylib-zig (c) Nikolas Wipper 2023
const std = @import("std");
const rl = @import("raylib");
const gui = @import("raygui");

const assert = std.debug.assert;

pub fn main() anyerror!void {
    // Initialization
    //--------------------------------------------------------------------------------------
    const screenWidth = 1000;
    const screenHeight = 720;

    rl.setConfigFlags(.{ .vsync_hint = true, .msaa_4x_hint = true, .window_resizable = true });
    rl.initWindow(screenWidth, screenHeight, "othello");
    defer rl.closeWindow(); // Close window and OpenGL context

    rl.setTargetFPS(60); // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------

    const main_board_pos: Vec2 = .{ .x = 20, .y = 20 };
    const main_board_size: f32 = 640;

    var board = init_board;
    var nextcol: Color = .white;
    var showhelpers = true;

    // Main game loop
    while (!rl.windowShouldClose()) { // Detect window close button or ESC key
        const clicked_square: ?V2i = sq: {
            if (rl.isMouseButtonPressed(rl.MouseButton.mouse_button_left)) {
                const p = Vec2.multiply(
                    Vec2.subtract(rl.getMousePosition(), main_board_pos),
                    .{ .x = 8.0 / main_board_size, .y = 8.0 / main_board_size },
                );
                if (p.x >= 0 and p.x < 8 and p.y >= 0 and p.y < 8)
                    break :sq .{ @intFromFloat(p.x), @intFromFloat(p.y) };
            }
            break :sq null;
        };

        if (clicked_square) |sq| play: {
            board = play(board, sq, nextcol) catch break :play;
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
        drawPawn(.{ .x = 815, .y = 65 }, 20, nextcol);
        _ = gui.guiCheckBox(.{ .x = 700, .y = 500, .width = 20, .height = 20 }, "show helpers", &showhelpers);
    }
}

fn addmul(p: Vec2, k: f32, d: Vec2) Vec2 {
    return Vec2.add(p, Vec2.multiply(d, .{ .x = k, .y = k }));
}
fn drawBoard(b: Board, pos: Vec2, size: f32) void {
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

fn drawPawn(pos: Vec2, radius: f32, col: Color) void {
    const rgb = switch (col) {
        .empty => return,
        .white => rl.Color.beige,
        .black => rl.Color.dark_brown,
    };

    rl.drawCircleV(pos, radius, rgb);
    rl.drawCircleLinesV(pos, radius, rl.Color.black);
}

fn drawHelper(b: Board, nextcol: Color, pos: Vec2, size: f32) void {
    const pos0 = Vec2.add(pos, .{ .x = size / 16, .y = size / 16 });

    const valids = computeValidSquares(b, nextcol);
    for (0..8) |y| {
        for (0..8) |x| {
            if (valids & Board.bitmask(x, y) == 0) continue;
            rl.drawCircleLinesV(addmul(pos0, size / 8, .{ .x = @floatFromInt(x), .y = @floatFromInt(y) }), size / 20, rl.Color.red);
        }
    }
}

//  ---- board

const Color = enum { empty, black, white };
const Vec2 = rl.Vector2;
const Board = struct {
    occupied: u64,
    color: u64,

    fn bitmask(x: anytype, y: anytype) u64 {
        return @as(u64, 1) << @intCast(y * 8 + x);
    }
    fn get(b: @This(), x: anytype, y: anytype) Color {
        assert(x < 8 and y < 8);
        const bit: u64 = bitmask(x, y);
        if ((b.occupied & bit) == 0) return .empty;
        return if ((b.color & bit) == 0) .black else .white;
    }
    fn set(b: *@This(), x: anytype, y: anytype, c: Color) void {
        assert(x < 8 and y < 8);
        const bit: u64 = bitmask(x, y);
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
};

const init_board: Board = init: {
    var b: Board = .{ .color = 0, .occupied = 0 };
    b.set(3, 3, .white);
    b.set(4, 4, .white);
    b.set(3, 4, .black);
    b.set(4, 3, .black);
    break :init b;
};

const V2i = [2]i8; //@Vector(2, i8); ..selfhosted
fn add(a: V2i, b: V2i) V2i {
    return .{ a[0] + b[0], a[1] + b[1] };
}
fn inBounds(p: V2i, min: V2i, max: V2i) bool {
    // @reduce(.And, @min(@max(p, V2i{ 0, 0 }), V2i{ 7, 7 }) == p))
    return p[0] >= min[0] and p[0] <= max[0] and p[1] >= min[1] and p[1] <= max[1];
}

const compass_dirs: []const V2i = &.{
    .{ 1, 0 }, .{ -1, 0 }, .{ 0, 1 },  .{ 0, -1 },
    .{ 1, 1 }, .{ -1, 1 }, .{ 1, -1 }, .{ -1, -1 },
};

fn computeValidSquares(b: Board, col: Color) u64 {
    assert(col != .empty);
    var valid: u64 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            if (b.get(x, y) != .empty) continue;

            const ok: bool = loop: for (compass_dirs) |d| {
                var p: V2i = .{ @intCast(x), @intCast(y) };
                var has_oppo = false;
                while (true) {
                    p = add(p, d);
                    if (!inBounds(p, V2i{ 0, 0 }, V2i{ 7, 7 })) continue :loop;
                    const sq = b.get(p[0], p[1]);
                    if (sq == .empty) continue :loop;
                    if (sq == col and !has_oppo) continue :loop;
                    if (sq == col and has_oppo) break :loop true;
                    if (sq != col) has_oppo = true;
                }
            } else false;

            if (ok) {
                valid |= Board.bitmask(x, y);
            }
        }
    }
    return valid;
}

fn play(b: Board, p: V2i, col: Color) !Board {
    assert(col != .empty);
    const valid = computeValidSquares(b, col);
    const bit = Board.bitmask(p[0], p[1]);
    if (valid & bit == 0) return error.invalid;
    var b1 = b;
    b1.set(p[0], p[1], col);

    loop: for (compass_dirs) |d| {
        var p1 = p;
        while (true) {
            p1 = add(p1, d);
            if (!inBounds(p1, V2i{ 0, 0 }, V2i{ 7, 7 })) continue :loop;
            const sq = b.get(p1[0], p1[1]);
            if (sq == .empty) continue :loop;
            if (sq == col) break;
        }

        p1 = p;
        while (true) {
            p1 = add(p1, d);
            assert(inBounds(p1, V2i{ 0, 0 }, V2i{ 7, 7 }));
            const sq = b.get(p1[0], p1[1]);
            assert(sq != .empty);
            if (sq == col) continue :loop;
            b1.set(p1[0], p1[1], col);
        }
    }
    return b1;
}
