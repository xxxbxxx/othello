// raylib-zig (c) Nikolas Wipper 2023

const rl = @import("raylib");
const gui = @import("raygui");

const Vec2 = rl.Vector2;

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
        drawBoard(.{ .x = 20, .y = 20 }, 640);

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
fn drawBoard(pos: Vec2, size: f32) void {
    const rect: rl.Rectangle = .{ .x = pos.x, .y = pos.y, .width = size, .height = size };
    rl.drawRectangleRounded(rect, 0.1, 7, rl.Color.dark_green);

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

    rl.drawRectangleRoundedLinesEx(rect, 0.1, 7, 5.0, rl.Color.black);

    //rl.drawCircle(200, 200, 100, rl.Color.red);
}
