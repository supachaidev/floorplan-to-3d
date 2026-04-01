/**
 * 2D Polygon Editor
 *
 * Interactions:
 *   Room corners:  drag to move | double-click edge to add point | right-click corner to delete
 *   Door hinge:    drag to move position | right-click to delete door
 *   Door rotation: drag diamond handle or scroll wheel over door area
 *   Add Door:      click toolbar button, then click canvas to place
 *   Room body:     double-click to rename | right-click to delete room
 *   Add Room:      click toolbar button, click points to define polygon, click first point or Enter to close
 *   Merge Rooms:   click toolbar button, click two rooms to merge into one (convex hull)
 *   Wall endpoints: drag to move | right-click to delete wall
 */
(function () {
    const canvas = document.getElementById("canvas-2d");
    const ctx = canvas.getContext("2d");

    let bgImage = null;
    let floorplanData = null;
    let rooms = [];
    let doors = [];
    let walls = [];            // { id, start: {x,y}, end: {x,y}, thickness }
    let dragState = null;
    let hoveredPoint = null;   // { roomIdx, ptIdx } | { doorIdx, rotate? } | { wallIdx, endpoint }
    let hoveredEdge = null;    // { roomIdx, edgeIdx, t, sx, sy } - for add-point preview
    let addDoorMode = false;
    let addWallMode = false;       // false | "drawing" (actively placing vertices)
    let addWallPoints = [];        // array of {x, y} in data coords — vertices being placed
    let addWallMousePos = null;    // current mouse position in screen coords for preview line
    let mergeRoomMode = false;     // false | { first: roomIdx } (waiting for second room click)
    let mergeHighlight = -1;       // roomIdx under mouse during merge mode

    const HANDLE_RADIUS = 6;
    const ROTATE_HANDLE_RADIUS = 7;
    const EDGE_HIT_DIST = 10;

    const ROOM_COLORS = {
        bedroom: "rgba(100, 149, 237, 0.25)",
        bathroom: "rgba(0, 206, 209, 0.25)",
        kitchen: "rgba(255, 165, 0, 0.25)",
        living: "rgba(144, 238, 144, 0.25)",
        dining: "rgba(255, 200, 100, 0.25)",
        hallway: "rgba(180, 180, 200, 0.2)",
        closet: "rgba(160, 140, 180, 0.2)",
        balcony: "rgba(120, 200, 200, 0.2)",
        other: "rgba(200, 200, 200, 0.2)",
    };
    const ROOM_BORDERS = {
        bedroom: "#6495ED", bathroom: "#00CED1", kitchen: "#FFA500",
        living: "#90EE90", dining: "#FFC864", hallway: "#B4B4C8",
        closet: "#A08CB4", balcony: "#78C8C8", other: "#aaa",
    };

    // ── coordinate helpers ──────────────────────────────────────────
    // The coordinate system maps meter-based polygon coordinates to the
    // same screen rectangle as the background image, so they align.
    let imgRect = { x: 0, y: 0, w: 1, h: 1 };

    function resizeCanvas() {
        const p = canvas.parentElement;
        canvas.width = p.clientWidth;
        canvas.height = p.clientHeight;
        computeImageRect();
        draw();
    }

    function computeImageRect() {
        const pad = 40, cw = canvas.width - pad * 2, ch = canvas.height - pad * 2;
        if (!bgImage) { imgRect = { x: pad, y: pad, w: cw, h: ch }; return; }
        const ia = bgImage.width / bgImage.height, ca = cw / ch;
        let dw, dh;
        if (ia > ca) { dw = cw; dh = cw / ia; } else { dh = ch; dw = ch * ia; }
        imgRect = { x: pad, y: pad, w: dw, h: dh };
    }

    function toScreen(pt) {
        // Map meter coordinates to image screen rect
        const bx = rooms._boundsX || 1, by = rooms._boundsY || 1;
        return {
            x: imgRect.x + (pt.x / bx) * imgRect.w,
            y: imgRect.y + (pt.y / by) * imgRect.h,
        };
    }
    function fromScreen(sx, sy) {
        const bx = rooms._boundsX || 1, by = rooms._boundsY || 1;
        return {
            x: ((sx - imgRect.x) / imgRect.w) * bx,
            y: ((sy - imgRect.y) / imgRect.h) * by,
        };
    }
    function getScale() {
        // Average scale factor for uniform-size elements (handles, door arcs)
        const bx = rooms._boundsX || 1, by = rooms._boundsY || 1;
        return Math.min(imgRect.w / bx, imgRect.h / by);
    }

    function getDoorRotateHandle(door) {
        const sp = toScreen(door.position), s = getScale();
        const r = (door.width || 0.9) * s, a = (door.angle || 0) * Math.PI / 180;
        return { x: sp.x + r * Math.cos(a), y: sp.y + r * Math.sin(a) };
    }

    // ── data ────────────────────────────────────────────────────────
    let isVlmSchema = false;

    function loadImage(url, floorplan) {
        const img = new Image();
        img.onload = () => { bgImage = img; floorplanData = floorplan; buildWorkingCopy(); computeImageRect(); resizeCanvas(); };
        img.src = url;
    }

    function buildWorkingCopy() {
        if (!floorplanData) return;

        // Detect VLM wall-first schema vs OpenCV room-polygon schema
        isVlmSchema = !!floorplanData.walls;

        if (isVlmSchema) {
            buildVlmWorkingCopy();
        } else {
            buildOpenCvWorkingCopy();
        }
    }

    function buildVlmWorkingCopy() {
        const vlmRooms = floorplanData.rooms || [];

        // VLM coordinates are normalized 0.0–1.0 fractions of image dimensions.
        // Bounds are 1.0 so toScreen maps directly to the displayed image rect.
        let boundsX = 1.0;
        let boundsY = 1.0;

        // Convert VLM rooms to editor format (polygon with {x,y} objects)
        // Coordinates are already in pixels — use them directly.
        rooms = vlmRooms.map(r => {
            const poly = (r.floor_polygon || []).map(p => ({
                x: p[0],
                y: p[1],
            }));
            // Classify room type from label
            const label = r.label || "Room";
            const typeLower = label.toLowerCase();
            let type = "other";
            if (typeLower.includes("bed")) type = "bedroom";
            else if (typeLower.includes("bath") || typeLower.includes("toilet")) type = "bathroom";
            else if (typeLower.includes("kitchen")) type = "kitchen";
            else if (typeLower.includes("living")) type = "living";
            else if (typeLower.includes("dining")) type = "dining";
            else if (typeLower.includes("hall") || typeLower.includes("corridor")) type = "hallway";
            else if (typeLower.includes("closet") || typeLower.includes("storage")) type = "closet";
            else if (typeLower.includes("balcon")) type = "balcony";
            return {
                id: r.id,
                label: label,
                type: type,
                height: 3.0,
                polygon: poly,
            };
        });

        // Convert VLM openings to editor door format
        // All coordinates stay in pixels to match the image.
        const openings = floorplanData.openings || [];
        const wallMap = {};
        for (const w of (floorplanData.walls || [])) wallMap[w.id] = w;

        doors = openings.filter(o => o.type === "door").map(o => {
            const wall = wallMap[o.wall_id];
            if (!wall) return null;
            // Calculate door position along wall (in pixels)
            const wx = wall.start[0], wy = wall.start[1];
            const dx = wall.end[0] - wx, dy = wall.end[1] - wy;
            const wLen = Math.sqrt(dx * dx + dy * dy);
            const t = wLen > 0 ? (o.position || 0) / wLen : 0.5;
            const posX = wx + dx * t;
            const posY = wy + dy * t;
            const angle = Math.atan2(dy, dx) * 180 / Math.PI;
            return {
                id: o.id,
                position: { x: posX, y: posY },
                width: o.width || 90,
                angle: angle,
                connects: [],
            };
        }).filter(Boolean);

        // Load VLM walls into editor
        walls = (floorplanData.walls || []).map(w => ({
            id: w.id,
            start: { x: w.start[0], y: w.start[1] },
            end: { x: w.end[0], y: w.end[1] },
            thickness: w.thickness || 0.02,
        }));

        rooms._boundsX = boundsX || 1;
        rooms._boundsY = boundsY || 1;
    }

    function buildOpenCvWorkingCopy() {
        const fp = floorplanData.floorplan;

        // Use full image dimensions (meters) as bounds so polygons align
        // with the background image. If not provided by API, compute from
        // the image dimensions using the same formula as the backend
        // (longest side = 15 meters).
        let boundsX = fp.image_width_m || 0;
        let boundsY = fp.image_height_m || 0;
        if ((!boundsX || !boundsY) && bgImage) {
            const maxDim = Math.max(bgImage.width, bgImage.height);
            const mpp = 15.0 / maxDim;
            boundsX = bgImage.width * mpp;
            boundsY = bgImage.height * mpp;
        }
        if (!boundsX || !boundsY) {
            for (const r of fp.rooms)
                for (const p of r.polygon) {
                    boundsX = Math.max(boundsX, p.x);
                    boundsY = Math.max(boundsY, p.y);
                }
            for (const d of (fp.doors || [])) {
                boundsX = Math.max(boundsX, d.position.x + (d.width || 0.9));
                boundsY = Math.max(boundsY, d.position.y + (d.width || 0.9));
            }
        }

        rooms = fp.rooms.map(r => ({
            ...r, polygon: r.polygon.map(p => ({ x: p.x, y: p.y })),
        }));
        doors = (fp.doors || []).map(d => ({
            ...d, position: { x: d.position.x, y: d.position.y },
        }));
        walls = []; // OpenCV schema derives walls from room polygons
        rooms._boundsX = boundsX || 1;
        rooms._boundsY = boundsY || 1;
    }

    // ── hit-testing ─────────────────────────────────────────────────

    /** Distance from point (px,py) to segment (ax,ay)-(bx,by), plus parameter t. */
    function pointToSegment(px, py, ax, ay, bx, by) {
        const dx = bx - ax, dy = by - ay;
        const lenSq = dx * dx + dy * dy;
        let t = lenSq === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
        const cx = ax + t * dx, cy = ay + t * dy;
        return { dist: Math.hypot(px - cx, py - cy), t, cx, cy };
    }

    /** Find the nearest polygon edge within EDGE_HIT_DIST, returns { roomIdx, edgeIdx, t, sx, sy } or null. */
    function findEdge(mx, my) {
        let best = null, bestDist = EDGE_HIT_DIST;
        for (let rIdx = 0; rIdx < rooms.length; rIdx++) {
            const poly = rooms[rIdx].polygon;
            for (let i = 0; i < poly.length; i++) {
                const a = toScreen(poly[i]);
                const b = toScreen(poly[(i + 1) % poly.length]);
                const { dist, t, cx, cy } = pointToSegment(mx, my, a.x, a.y, b.x, b.y);
                // Ignore hits very close to existing vertices (those are handle hits)
                if (t < 0.1 || t > 0.9) continue;
                if (dist < bestDist) {
                    bestDist = dist;
                    best = { roomIdx: rIdx, edgeIdx: i, t, sx: cx, sy: cy };
                }
            }
        }
        return best;
    }

    function findHandle(mx, my) {
        // Door rotation handles (highest priority)
        for (let d = 0; d < doors.length; d++) {
            const rh = getDoorRotateHandle(doors[d]);
            if (Math.hypot(mx - rh.x, my - rh.y) < ROTATE_HANDLE_RADIUS + 5)
                return { doorIdx: d, rotate: true };
        }
        // Wall endpoint handles
        for (let w = 0; w < walls.length; w++) {
            const ss = toScreen(walls[w].start);
            if (Math.hypot(mx - ss.x, my - ss.y) < HANDLE_RADIUS + 4)
                return { wallIdx: w, endpoint: "start" };
            const se = toScreen(walls[w].end);
            if (Math.hypot(mx - se.x, my - se.y) < HANDLE_RADIUS + 4)
                return { wallIdx: w, endpoint: "end" };
        }
        // Room polygon corners
        for (let r = 0; r < rooms.length; r++) {
            for (let p = 0; p < rooms[r].polygon.length; p++) {
                const sp = toScreen(rooms[r].polygon[p]);
                if (Math.hypot(mx - sp.x, my - sp.y) < HANDLE_RADIUS + 4)
                    return { roomIdx: r, ptIdx: p };
            }
        }
        // Door hinge handles
        for (let d = 0; d < doors.length; d++) {
            const sp = toScreen(doors[d].position);
            if (Math.hypot(mx - sp.x, my - sp.y) < HANDLE_RADIUS + 4)
                return { doorIdx: d, rotate: false };
        }
        return null;
    }

    function findDoorUnderMouse(mx, my) {
        const s = getScale();
        for (let d = 0; d < doors.length; d++) {
            const sp = toScreen(doors[d].position);
            if (Math.hypot(mx - sp.x, my - sp.y) < (doors[d].width || 0.9) * s + 10) return d;
        }
        return -1;
    }

    // ── polygon / merge helpers ────────────────────────────────────

    /** Snap a data-coordinate point to nearby existing room corners or in-progress points. */
    function snapToNearby(mx, my, dataPt) {
        const SNAP_DIST = 10; // screen pixels
        // Check existing room polygon corners
        for (const room of rooms) {
            for (const p of room.polygon) {
                const sp = toScreen(p);
                if (Math.hypot(mx - sp.x, my - sp.y) < SNAP_DIST)
                    return { x: p.x, y: p.y };
            }
        }
        // Check already-placed points in current polygon
        for (const p of addWallPoints) {
            const sp = toScreen(p);
            if (Math.hypot(mx - sp.x, my - sp.y) < SNAP_DIST)
                return { x: p.x, y: p.y };
        }
        return dataPt;
    }

    /** Close the polygon being drawn and create a new room from it. */
    function finishAddWallPolygon() {
        if (addWallPoints.length < 3) { addWallPoints = []; addWallMode = false; return; }
        const nextId = "room_user_" + (rooms.length + 1);
        rooms.push({
            id: nextId,
            label: "Room " + (rooms.length + 1),
            type: "other",
            height: 3.0,
            polygon: addWallPoints.map(p => ({ x: p.x, y: p.y })),
        });
        addWallPoints = [];
        addWallMode = false;
        canvas.style.cursor = "default";
    }

    /** Find which room polygon contains the screen point (mx, my). */
    function findRoomUnderMouse(mx, my) {
        for (let r = rooms.length - 1; r >= 0; r--) {
            const poly = rooms[r].polygon;
            if (poly.length < 3) continue;
            if (pointInPolygon(mx, my, poly.map(p => toScreen(p)))) return r;
        }
        return -1;
    }

    /** Point-in-polygon test (screen coords). */
    function pointInPolygon(px, py, screenPoly) {
        let inside = false;
        for (let i = 0, j = screenPoly.length - 1; i < screenPoly.length; j = i++) {
            const xi = screenPoly[i].x, yi = screenPoly[i].y;
            const xj = screenPoly[j].x, yj = screenPoly[j].y;
            if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi)
                inside = !inside;
        }
        return inside;
    }

    /** Merge two rooms by combining their polygons using convex hull. */
    function mergeRooms(idxA, idxB) {
        const a = rooms[idxA], b = rooms[idxB];
        const allPts = [...a.polygon, ...b.polygon];
        const hull = convexHull(allPts);

        // Keep the first room's metadata, update polygon
        a.polygon = hull;
        a.label = a.label + " + " + b.label;
        // Remove the second room
        const removeIdx = Math.max(idxA, idxB);
        const keepIdx = Math.min(idxA, idxB);
        rooms.splice(removeIdx, 1);
        // If we removed before the kept index, adjust
        // (not needed since we always splice the higher index)
    }

    /** Compute convex hull of a set of {x, y} points (Andrew's monotone chain). */
    function convexHull(points) {
        const pts = points.slice().sort((a, b) => a.x - b.x || a.y - b.y);
        if (pts.length <= 2) return pts;
        const cross = (O, A, B) => (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
        const lower = [];
        for (const p of pts) {
            while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
            lower.push(p);
        }
        const upper = [];
        for (let i = pts.length - 1; i >= 0; i--) {
            while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], pts[i]) <= 0) upper.pop();
            upper.push(pts[i]);
        }
        upper.pop(); lower.pop();
        return lower.concat(upper);
    }

    // ── drawing ─────────────────────────────────────────────────────
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (bgImage) {
            ctx.globalAlpha = 0.3;
            ctx.drawImage(bgImage, imgRect.x, imgRect.y, imgRect.w, imgRect.h);
            ctx.globalAlpha = 1.0;
        }
        if (!rooms.length) return;

        rooms.forEach((room, rIdx) => {
            if (room.polygon.length < 2) return;
            const color = ROOM_COLORS[room.type] || ROOM_COLORS.other;
            const border = ROOM_BORDERS[room.type] || ROOM_BORDERS.other;

            // Fill + stroke
            ctx.beginPath();
            const f = toScreen(room.polygon[0]);
            ctx.moveTo(f.x, f.y);
            for (let i = 1; i < room.polygon.length; i++) { const p = toScreen(room.polygon[i]); ctx.lineTo(p.x, p.y); }
            ctx.closePath();
            ctx.fillStyle = color; ctx.fill();
            ctx.strokeStyle = border; ctx.lineWidth = 2; ctx.stroke();

            // Label
            const cx = room.polygon.reduce((s, p) => s + p.x, 0) / room.polygon.length;
            const cy = room.polygon.reduce((s, p) => s + p.y, 0) / room.polygon.length;
            const ctr = toScreen({ x: cx, y: cy });
            ctx.fillStyle = "#fff"; ctx.font = "13px -apple-system, sans-serif";
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText(room.label, ctr.x, ctr.y);

            // Edge midpoint "+" indicators (only for hovered room or hovered edge)
            if (hoveredEdge && hoveredEdge.roomIdx === rIdx) {
                // Show small "+" at the projected point on the edge
                const he = hoveredEdge;
                ctx.beginPath();
                ctx.arc(he.sx, he.sy, 5, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(255,255,255,0.9)";
                ctx.fill();
                ctx.strokeStyle = border;
                ctx.lineWidth = 1.5;
                ctx.stroke();
                // Plus sign
                ctx.beginPath();
                ctx.moveTo(he.sx - 3, he.sy); ctx.lineTo(he.sx + 3, he.sy);
                ctx.moveTo(he.sx, he.sy - 3); ctx.lineTo(he.sx, he.sy + 3);
                ctx.strokeStyle = border;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Corner handles
            room.polygon.forEach((pt, pIdx) => {
                const sp = toScreen(pt);
                const isHovered = hoveredPoint && hoveredPoint.roomIdx === rIdx && hoveredPoint.ptIdx === pIdx;
                ctx.beginPath();
                ctx.arc(sp.x, sp.y, isHovered ? HANDLE_RADIUS + 2 : HANDLE_RADIUS, 0, Math.PI * 2);
                ctx.fillStyle = isHovered ? "#fff" : border;
                ctx.fill();
                ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.stroke();

                // Show "x" hint on right-click-deletable points (when hovered, 4+ vertices)
                if (isHovered && room.polygon.length > 3) {
                    ctx.fillStyle = "#000"; ctx.font = "bold 9px sans-serif";
                    ctx.textAlign = "center"; ctx.textBaseline = "middle";
                    ctx.fillText("x", sp.x, sp.y);
                }
            });
        });

        drawDoors();
        drawWalls();

        if (addWallMode) {
            // Draw in-progress polygon
            if (addWallPoints.length > 0) {
                ctx.beginPath();
                const first = toScreen(addWallPoints[0]);
                ctx.moveTo(first.x, first.y);
                for (let i = 1; i < addWallPoints.length; i++) {
                    const p = toScreen(addWallPoints[i]);
                    ctx.lineTo(p.x, p.y);
                }
                // Preview line to mouse
                if (addWallMousePos) {
                    ctx.lineTo(addWallMousePos.x, addWallMousePos.y);
                }
                ctx.strokeStyle = "rgba(255,200,50,0.8)";
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.stroke();
                ctx.setLineDash([]);

                // Fill preview if 3+ points
                if (addWallPoints.length >= 3) {
                    ctx.beginPath();
                    ctx.moveTo(first.x, first.y);
                    for (let i = 1; i < addWallPoints.length; i++) {
                        const p = toScreen(addWallPoints[i]);
                        ctx.lineTo(p.x, p.y);
                    }
                    ctx.closePath();
                    ctx.fillStyle = "rgba(255,200,50,0.1)";
                    ctx.fill();
                }

                // Draw vertex handles
                addWallPoints.forEach((pt, i) => {
                    const sp = toScreen(pt);
                    ctx.beginPath();
                    ctx.arc(sp.x, sp.y, HANDLE_RADIUS, 0, Math.PI * 2);
                    ctx.fillStyle = i === 0 ? "#FF6B6B" : "#FFC832";
                    ctx.fill();
                    ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.stroke();
                });

                // Close hint on first point
                if (addWallPoints.length >= 3 && addWallMousePos) {
                    const d = Math.hypot(addWallMousePos.x - first.x, addWallMousePos.y - first.y);
                    if (d < 20) {
                        ctx.beginPath();
                        ctx.arc(first.x, first.y, 12, 0, Math.PI * 2);
                        ctx.strokeStyle = "rgba(255,107,107,0.8)";
                        ctx.lineWidth = 2;
                        ctx.stroke();
                    }
                }
            }

            ctx.fillStyle = "rgba(255,200,50,0.85)";
            ctx.font = "bold 13px -apple-system, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            const n = addWallPoints.length;
            let msg;
            if (n === 0) msg = "Click to place first vertex  (Esc to cancel)";
            else if (n < 3) msg = `${n} point${n > 1 ? "s" : ""} placed — keep clicking to add more  (Esc to cancel)`;
            else msg = `${n} points — click first point (red) or press Enter to close  (Esc to cancel)`;
            ctx.fillText(msg, canvas.width / 2, 14);
        }

        if (mergeRoomMode) {
            // Highlight room under mouse
            if (mergeHighlight >= 0 && mergeHighlight < rooms.length) {
                const room = rooms[mergeHighlight];
                ctx.beginPath();
                const f = toScreen(room.polygon[0]);
                ctx.moveTo(f.x, f.y);
                for (let i = 1; i < room.polygon.length; i++) { const p = toScreen(room.polygon[i]); ctx.lineTo(p.x, p.y); }
                ctx.closePath();
                ctx.strokeStyle = "#FFD700";
                ctx.lineWidth = 3;
                ctx.setLineDash([6, 3]);
                ctx.stroke();
                ctx.setLineDash([]);
            }
            // Highlight already-selected first room
            if (mergeRoomMode.first !== undefined && mergeRoomMode.first < rooms.length) {
                const room = rooms[mergeRoomMode.first];
                ctx.beginPath();
                const f = toScreen(room.polygon[0]);
                ctx.moveTo(f.x, f.y);
                for (let i = 1; i < room.polygon.length; i++) { const p = toScreen(room.polygon[i]); ctx.lineTo(p.x, p.y); }
                ctx.closePath();
                ctx.fillStyle = "rgba(255,215,0,0.2)";
                ctx.fill();
                ctx.strokeStyle = "#FFD700";
                ctx.lineWidth = 3;
                ctx.stroke();
            }

            ctx.fillStyle = "rgba(255,215,0,0.85)";
            ctx.font = "bold 13px -apple-system, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            const msg = mergeRoomMode.first !== undefined
                ? "Now click the second room to merge  (Esc to cancel)"
                : "Click the first room to merge  (Esc to cancel)";
            ctx.fillText(msg, canvas.width / 2, 14);
        }

        if (addDoorMode) {
            ctx.fillStyle = "rgba(255,107,107,0.85)";
            ctx.font = "bold 13px -apple-system, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            ctx.fillText("Click to place door  (Esc to cancel)", canvas.width / 2, 14);
        }
    }

    function drawDoors() {
        const scale = getScale();
        doors.forEach((door, dIdx) => {
            const sp = toScreen(door.position);
            const rPx = (door.width || 0.9) * scale;
            const mid = (door.angle || 0) * Math.PI / 180;
            const cA = mid - Math.PI / 4, oA = mid + Math.PI / 4;

            // Filled sector
            ctx.beginPath(); ctx.moveTo(sp.x, sp.y);
            ctx.arc(sp.x, sp.y, rPx, cA, oA); ctx.closePath();
            ctx.fillStyle = "rgba(255,107,107,0.1)"; ctx.fill();

            // Arc (dashed)
            ctx.beginPath(); ctx.arc(sp.x, sp.y, rPx, cA, oA);
            ctx.strokeStyle = "#FF6B6B"; ctx.lineWidth = 2;
            ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([]);

            // Closed edge (solid)
            ctx.beginPath(); ctx.moveTo(sp.x, sp.y);
            ctx.lineTo(sp.x + rPx * Math.cos(cA), sp.y + rPx * Math.sin(cA));
            ctx.strokeStyle = "#FF6B6B"; ctx.lineWidth = 2; ctx.stroke();

            // Open edge (dashed)
            ctx.beginPath(); ctx.moveTo(sp.x, sp.y);
            ctx.lineTo(sp.x + rPx * Math.cos(oA), sp.y + rPx * Math.sin(oA));
            ctx.strokeStyle = "rgba(255,107,107,0.4)"; ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);

            // Hinge handle
            const hH = hoveredPoint && hoveredPoint.doorIdx === dIdx && !hoveredPoint.rotate;
            ctx.beginPath(); ctx.arc(sp.x, sp.y, hH ? 6 : 4, 0, Math.PI * 2);
            ctx.fillStyle = hH ? "#fff" : "#FF6B6B"; ctx.fill();
            ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.stroke();

            // Rotation handle (diamond)
            const rh = getDoorRotateHandle(door);
            const rH = hoveredPoint && hoveredPoint.doorIdx === dIdx && hoveredPoint.rotate;
            const sz = rH ? ROTATE_HANDLE_RADIUS + 2 : ROTATE_HANDLE_RADIUS;
            ctx.save(); ctx.translate(rh.x, rh.y); ctx.rotate(Math.PI / 4);
            ctx.beginPath(); ctx.rect(-sz / 2, -sz / 2, sz, sz);
            ctx.fillStyle = rH ? "#fff" : "#FF9B9B"; ctx.fill();
            ctx.strokeStyle = "#CC4444"; ctx.lineWidth = 1.5; ctx.stroke();
            ctx.restore();

            if (rH) {
                ctx.beginPath(); ctx.arc(rh.x, rh.y, sz + 6, 0, Math.PI * 1.5);
                ctx.strokeStyle = "rgba(255,255,255,0.6)"; ctx.lineWidth = 1.5; ctx.stroke();
            }

            // Label
            ctx.fillStyle = "#FF6B6B"; ctx.font = "10px -apple-system, sans-serif";
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText("Door", sp.x + rPx * 0.45 * Math.cos(mid), sp.y + rPx * 0.45 * Math.sin(mid));
        });
    }

    /** Compute the 4 screen-space corners of a wall rectangle. */
    function wallRectCorners(wall) {
        const s = toScreen(wall.start);
        const e = toScreen(wall.end);
        const dx = e.x - s.x, dy = e.y - s.y;
        const len = Math.hypot(dx, dy);
        if (len < 0.1) return null;
        // Perpendicular offset in screen pixels from thickness
        const scale = getScale();
        const halfT = (wall.thickness || 0.02) * scale / 2;
        const nx = (-dy / len) * halfT;
        const ny = (dx / len) * halfT;
        return [
            { x: s.x + nx, y: s.y + ny },
            { x: e.x + nx, y: e.y + ny },
            { x: e.x - nx, y: e.y - ny },
            { x: s.x - nx, y: s.y - ny },
        ];
    }

    function drawWalls() {
        walls.forEach((wall, wIdx) => {
            const corners = wallRectCorners(wall);
            if (!corners) return;

            // Filled rectangle
            ctx.beginPath();
            ctx.moveTo(corners[0].x, corners[0].y);
            for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
            ctx.closePath();
            ctx.fillStyle = "rgba(255,200,50,0.25)";
            ctx.fill();
            ctx.strokeStyle = "rgba(255,200,50,0.7)";
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Center line (dashed)
            const s = toScreen(wall.start);
            const e = toScreen(wall.end);
            ctx.beginPath();
            ctx.moveTo(s.x, s.y);
            ctx.lineTo(e.x, e.y);
            ctx.strokeStyle = "rgba(255,200,50,0.4)";
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 3]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Start endpoint handle
            const hS = hoveredPoint && hoveredPoint.wallIdx === wIdx && hoveredPoint.endpoint === "start";
            ctx.beginPath();
            ctx.arc(s.x, s.y, hS ? HANDLE_RADIUS + 2 : HANDLE_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = hS ? "#fff" : "#FFC832";
            ctx.fill();
            ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.stroke();

            // End endpoint handle
            const hE = hoveredPoint && hoveredPoint.wallIdx === wIdx && hoveredPoint.endpoint === "end";
            ctx.beginPath();
            ctx.arc(e.x, e.y, hE ? HANDLE_RADIUS + 2 : HANDLE_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = hE ? "#fff" : "#FFC832";
            ctx.fill();
            ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.stroke();

            // Wall label at midpoint
            const mx = (s.x + e.x) / 2, my = (s.y + e.y) / 2;
            ctx.fillStyle = "rgba(255,200,50,0.8)";
            ctx.font = "9px -apple-system, sans-serif";
            ctx.textAlign = "center"; ctx.textBaseline = "bottom";
            ctx.fillText(wall.id, mx, my - 4);
        });

    }

    // ── event handlers ──────────────────────────────────────────────

    canvas.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return; // left click only
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;

        if (addWallMode) {
            const pt = fromScreen(mx, my);
            const snap = { x: Math.round(pt.x * 10000) / 10000, y: Math.round(pt.y * 10000) / 10000 };

            // Snap to existing room corners or other placed points
            const snapped = snapToNearby(mx, my, snap);

            // If we have 3+ points and click near the first point, close the polygon
            if (addWallPoints.length >= 3) {
                const firstScreen = toScreen(addWallPoints[0]);
                if (Math.hypot(mx - firstScreen.x, my - firstScreen.y) < 12) {
                    finishAddWallPolygon();
                    draw();
                    return;
                }
            }

            addWallPoints.push(snapped);
            draw();
            return;
        }

        if (mergeRoomMode) {
            const rIdx = findRoomUnderMouse(mx, my);
            if (rIdx < 0) { draw(); return; }

            if (!mergeRoomMode.first && mergeRoomMode.first !== 0) {
                mergeRoomMode = { first: rIdx };
            } else if (rIdx !== mergeRoomMode.first) {
                mergeRooms(mergeRoomMode.first, rIdx);
                mergeRoomMode = false;
                mergeHighlight = -1;
                canvas.style.cursor = "default";
            }
            draw();
            return;
        }

        if (addDoorMode) {
            const pt = fromScreen(mx, my);
            const nextId = "d" + (doors.length + 1);
            doors.push({
                id: nextId,
                position: { x: Math.round(pt.x * 10000) / 10000, y: Math.round(pt.y * 10000) / 10000 },
                width: isVlmSchema ? 0.04 : 0.9,
                angle: 0,
                connects: [],
            });
            addDoorMode = false;
            canvas.style.cursor = "default";
            draw();
            return;
        }

        const handle = findHandle(mx, my);
        if (handle) {
            dragState = handle;
            canvas.style.cursor = "grabbing";
        }
    });

    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;

        if (addWallMode) {
            addWallMousePos = { x: mx, y: my };
            canvas.style.cursor = "crosshair";
            draw();
            return;
        }

        if (mergeRoomMode) {
            mergeHighlight = findRoomUnderMouse(mx, my);
            canvas.style.cursor = mergeHighlight >= 0 ? "pointer" : "crosshair";
            draw();
            return;
        }

        if (dragState) {
            if (dragState.rotate) {
                const door = doors[dragState.doorIdx];
                const sp = toScreen(door.position);
                const a = Math.atan2(my - sp.y, mx - sp.x);
                door.angle = Math.round(((a * 180 / Math.PI) % 360 + 360) % 360);
            } else if (dragState.wallIdx !== undefined) {
                const pt = fromScreen(mx, my);
                const w = walls[dragState.wallIdx];
                const ep = w[dragState.endpoint];
                ep.x = Math.round(pt.x * 10000) / 10000;
                ep.y = Math.round(pt.y * 10000) / 10000;
            } else if (dragState.doorIdx !== undefined) {
                const pt = fromScreen(mx, my);
                doors[dragState.doorIdx].position.x = Math.round(pt.x * 100) / 100;
                doors[dragState.doorIdx].position.y = Math.round(pt.y * 100) / 100;
            } else {
                const pt = fromScreen(mx, my);
                rooms[dragState.roomIdx].polygon[dragState.ptIdx].x = Math.round(pt.x * 100) / 100;
                rooms[dragState.roomIdx].polygon[dragState.ptIdx].y = Math.round(pt.y * 100) / 100;
            }
            hoveredEdge = null;
            draw();
        } else {
            const handle = findHandle(mx, my);
            hoveredPoint = handle;
            // Only look for edge hover when not over a handle
            hoveredEdge = handle ? null : findEdge(mx, my);

            if (addDoorMode) canvas.style.cursor = "crosshair";
            else if (handle && handle.rotate) canvas.style.cursor = "crosshair";
            else if (handle) canvas.style.cursor = "grab";
            else if (hoveredEdge) canvas.style.cursor = "copy";
            else canvas.style.cursor = "default";
            draw();
        }
    });

    canvas.addEventListener("mouseup", () => {
        dragState = null;
        canvas.style.cursor = "default";
    });

    canvas.addEventListener("mouseleave", () => {
        dragState = null; hoveredPoint = null; hoveredEdge = null;
        draw();
    });

    // Double-click on edge → add point | double-click inside room → rename
    canvas.addEventListener("dblclick", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;

        // If double-clicking a handle, ignore
        if (findHandle(mx, my)) return;

        const edge = findEdge(mx, my);
        if (edge) {
            const poly = rooms[edge.roomIdx].polygon;
            const a = poly[edge.edgeIdx];
            const b = poly[(edge.edgeIdx + 1) % poly.length];
            const newPt = {
                x: Math.round((a.x + (b.x - a.x) * edge.t) * 100) / 100,
                y: Math.round((a.y + (b.y - a.y) * edge.t) * 100) / 100,
            };
            // Insert after edgeIdx
            poly.splice(edge.edgeIdx + 1, 0, newPt);
            hoveredEdge = null;
            draw();
            return;
        }

        // Double-click inside room body → rename
        const rIdx = findRoomUnderMouse(mx, my);
        if (rIdx >= 0) {
            const room = rooms[rIdx];
            const newName = prompt("Rename room:", room.label);
            if (newName !== null && newName.trim() !== "") {
                room.label = newName.trim();
                // Update type based on new name
                const typeLower = room.label.toLowerCase();
                if (typeLower.includes("bed")) room.type = "bedroom";
                else if (typeLower.includes("bath") || typeLower.includes("toilet")) room.type = "bathroom";
                else if (typeLower.includes("kitchen")) room.type = "kitchen";
                else if (typeLower.includes("living")) room.type = "living";
                else if (typeLower.includes("dining")) room.type = "dining";
                else if (typeLower.includes("hall") || typeLower.includes("corridor")) room.type = "hallway";
                else if (typeLower.includes("closet") || typeLower.includes("storage")) room.type = "closet";
                else if (typeLower.includes("balcon")) room.type = "balcony";
                else room.type = "other";
                draw();
            }
        }
    });

    // Right-click on corner → delete point (min 3 vertices)
    // Right-click on door hinge → delete door
    // Right-click inside room (not on handle) → delete room
    canvas.addEventListener("contextmenu", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const handle = findHandle(mx, my);
        if (handle && handle.wallIdx !== undefined) {
            e.preventDefault();
            walls.splice(handle.wallIdx, 1);
            hoveredPoint = null;
            draw();
        } else if (handle && handle.doorIdx !== undefined && !handle.rotate) {
            e.preventDefault();
            doors.splice(handle.doorIdx, 1);
            hoveredPoint = null;
            draw();
        } else if (handle && handle.roomIdx !== undefined && handle.ptIdx !== undefined) {
            const poly = rooms[handle.roomIdx].polygon;
            if (poly.length > 3) {
                e.preventDefault();
                poly.splice(handle.ptIdx, 1);
                hoveredPoint = null;
                draw();
            }
        } else {
            // No handle hit — check if right-clicking inside a room body
            const rIdx = findRoomUnderMouse(mx, my);
            if (rIdx >= 0) {
                e.preventDefault();
                const room = rooms[rIdx];
                if (confirm(`Delete "${room.label}"?`)) {
                    rooms.splice(rIdx, 1);
                    draw();
                }
            }
        }
    });

    /** Find wall index whose center line is near screen point (mx, my). */
    function findWallUnderMouse(mx, my) {
        for (let w = 0; w < walls.length; w++) {
            const s = toScreen(walls[w].start);
            const e = toScreen(walls[w].end);
            const { dist } = pointToSegment(mx, my, s.x, s.y, e.x, e.y);
            const scale = getScale();
            const halfT = (walls[w].thickness || 0.02) * scale / 2;
            if (dist < halfT + 8) return w;
        }
        return -1;
    }

    // Scroll wheel → rotate doors / adjust wall thickness
    canvas.addEventListener("wheel", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;

        // Check wall first (thickness adjust)
        const wIdx = findWallUnderMouse(mx, my);
        if (wIdx >= 0) {
            e.preventDefault();
            const step = isVlmSchema ? 0.003 : 0.05;
            const min = isVlmSchema ? 0.005 : 0.05;
            walls[wIdx].thickness = Math.max(min, (walls[wIdx].thickness || 0.02) + (e.deltaY > 0 ? step : -step));
            walls[wIdx].thickness = Math.round(walls[wIdx].thickness * 1000) / 1000;
            draw();
            return;
        }

        const dIdx = findDoorUnderMouse(mx, my);
        if (dIdx >= 0) {
            e.preventDefault();
            doors[dIdx].angle = ((doors[dIdx].angle || 0) + (e.deltaY > 0 ? 15 : -15) + 360) % 360;
            draw();
        }
    }, { passive: false });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            if (addDoorMode) { addDoorMode = false; }
            if (addWallMode) { addWallMode = false; addWallPoints = []; addWallMousePos = null; }
            if (mergeRoomMode) { mergeRoomMode = false; mergeHighlight = -1; }
            canvas.style.cursor = "default";
            draw();
        }
        if (e.key === "Enter" && addWallMode && addWallPoints.length >= 3) {
            finishAddWallPolygon();
            draw();
        }
    });

    window.addEventListener("resize", resizeCanvas);

    // ── public API ──────────────────────────────────────────────────

    function resetPolygons() { buildWorkingCopy(); draw(); }

    function getFloorplanData() {
        if (!floorplanData) return null;
        const edited = JSON.parse(JSON.stringify(floorplanData));

        if (isVlmSchema) {
            // Update VLM schema walls with edited positions
            edited.walls = walls.map(w => ({
                id: w.id,
                start: [w.start.x, w.start.y],
                end: [w.end.x, w.end.y],
                thickness: w.thickness || 0.02,
            }));
            // Update VLM schema rooms with edited polygons (normalized 0-1 coords)
            edited.rooms = rooms.map((r, i) => {
                const orig = edited.rooms[i] || {};
                return {
                    ...orig,
                    id: r.id,
                    label: r.label,
                    floor_polygon: r.polygon.map(p => [
                        Math.round(p.x * 10000) / 10000,
                        Math.round(p.y * 10000) / 10000,
                    ]),
                };
            });
            // Export editor doors so the viewer can render them
            edited._editorDoors = doors.map(d => ({
                id: d.id,
                position: { x: d.position.x, y: d.position.y },
                width: d.width,
                angle: d.angle,
            }));
        } else {
            edited.floorplan.rooms = rooms.map(r => ({
                id: r.id, label: r.label, height: r.height, type: r.type,
                polygon: r.polygon.map(p => ({ x: Math.round(p.x * 100) / 100, y: Math.round(p.y * 100) / 100 })),
            }));
            edited.floorplan.doors = doors.map(d => ({
                id: d.id, width: d.width, angle: d.angle, connects: d.connects || [],
                position: { x: Math.round(d.position.x * 100) / 100, y: Math.round(d.position.y * 100) / 100 },
            }));
        }
        return edited;
    }

    function confirmAndRender3D() {
        const data = getFloorplanData();
        if (data) {
            window.viewer.render(data);
            const vt = document.getElementById("viewer-toolbar");
            if (vt) vt.style.display = "flex";
        }
    }

    function startAddDoor() {
        addWallMode = false;
        addWallPoints = [];
        addWallMousePos = null;
        mergeRoomMode = false;
        mergeHighlight = -1;
        addDoorMode = true;
        canvas.style.cursor = "crosshair";
        draw();
    }

    function startAddWall() {
        addDoorMode = false;
        mergeRoomMode = false;
        mergeHighlight = -1;
        addWallMode = "drawing";
        addWallPoints = [];
        addWallMousePos = null;
        canvas.style.cursor = "crosshair";
        draw();
    }

    function startMergeRooms() {
        addDoorMode = false;
        addWallMode = false;
        addWallPoints = [];
        addWallMousePos = null;
        mergeRoomMode = {};
        mergeHighlight = -1;
        canvas.style.cursor = "crosshair";
        draw();
    }

    window.editor = { loadImage, resetPolygons, confirmAndRender3D, getFloorplanData, startAddDoor, startAddWall, startMergeRooms };
})();
