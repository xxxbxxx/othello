// raylib-zig (c) Nikolas Wipper 2023
const std = @import("std");
const rl = @import("raylib");
const gui = @import("raygui");

const assert = std.debug.assert;

const Color = enum { empty, black, white };
const Vec2 = rl.Vector2;
const Board = struct {
    occupied: u64,
    color: u64,

    fn get(b: @This(), x: usize, y: usize) Color {
        assert(x < 8 and y < 8);
        const bit: u64 = @as(u64, 1) << @intCast(y * 8 + x);
        if ((b.occupied & bit) == 0) return .empty;
        return if ((b.color & bit) == 0) .black else .white;
    }
    fn set(b: *@This(), x: usize, y: usize, c: Color) void {
        assert(x < 8 and y < 8);
        const bit: u64 = @as(u64, 1) << @intCast(y * 8 + x);
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

    // Main game loop
    while (!rl.windowShouldClose()) { // Detect window close button or ESC key
        // Update
        //----------------------------------------------------------------------------------
        // TODO: Update your variables here
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.dark_brown);
        drawBoard(init_board, .{ .x = 20, .y = 20 }, 640);

        //            rl.drawCircle(200, 200, 100, rl.Color.red);
        rl.drawText("Congrats! You created your first window!", 190, 200, 20, rl.Color.light_gray);
        //----------------------------------------------------------------------------------

        if (gui.guiLabelButton(.{ .x = 10, .y = 10, .width = 100, .height = 100 }, "coucou") != 0) {
            //   draw_grid = !draw_grid;
        }
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
                const col = switch (b.get(x, y)) {
                    .empty => continue,
                    .white => rl.Color.beige,
                    .black => rl.Color.dark_brown,
                };

                rl.drawCircleV(addmul(pos0, size / 8, .{ .x = @floatFromInt(x), .y = @floatFromInt(y) }), size / 20, col);
                rl.drawCircleLinesV(addmul(pos0, size / 8, .{ .x = @floatFromInt(x), .y = @floatFromInt(y) }), size / 20, rl.Color.black);
            }
        }
    }

    rl.drawRectangleRoundedLinesEx(rect, 0.1, 7, 5.0, rl.Color.black);
}
