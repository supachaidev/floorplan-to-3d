/**
 * 2D Polygon Editor
 *
 * Interactions:
 *   Room corners:  drag to move | double-click edge to add point | right-click corner to delete
 *   Door hinge:    drag to move position
 *   Door rotation: drag diamond handle or scroll wheel over door area
 */
(function () {
    const canvas = document.getElementById("canvas-2d");
    const ctx = canvas.getContext("2d");

    let bgImage = null;
    let floorplanData = null;
    let rooms = [];
    let doors = [];
    let dragState = null;
    let hoveredPoint = null;   // { roomIdx, ptIdx } | { doorIdx, rotate? }
    let hoveredEdge = null;    // { roomIdx, edgeIdx, t, sx, sy } - for add-point preview
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
    function resizeCanvas() {
        const p = canvas.parentElement;
        canvas.width = p.clientWidth;
        canvas.height = p.clientHeight;
        draw();
    }

    function getScale() {
        const pad = 40, cw = canvas.width - pad * 2, ch = canvas.height - pad * 2;
        return Math.min(cw / (rooms._boundsX || 1), ch / (rooms._boundsY || 1));
    }
    function toScreen(pt) {
        const pad = 40, s = getScale();
        return { x: pad + pt.x * s, y: pad + pt.y * s };
    }
    function fromScreen(sx, sy) {
        const pad = 40, s = getScale();
        return { x: (sx - pad) / s, y: (sy - pad) / s };
    }

    function getDoorRotateHandle(door) {
        const sp = toScreen(door.position), s = getScale();
        const r = (door.width || 0.9) * s, a = (door.angle || 0) * Math.PI / 180;
        return { x: sp.x + r * Math.cos(a), y: sp.y + r * Math.sin(a) };
    }

    // ── data ────────────────────────────────────────────────────────
    function loadImage(url, floorplan) {
        const img = new Image();
        img.onload = () => { bgImage = img; floorplanData = floorplan; buildWorkingCopy(); resizeCanvas(); };
        img.src = url;
    }

    function buildWorkingCopy() {
        if (!floorplanData) return;
        let maxX = 0, maxY = 0;
        for (const r of floorplanData.floorplan.rooms)
            for (const p of r.polygon) { maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y); }
        for (const d of (floorplanData.floorplan.doors || []))
            { maxX = Math.max(maxX, d.position.x + (d.width || 0.9)); maxY = Math.max(maxY, d.position.y + (d.width || 0.9)); }

        rooms = floorplanData.floorplan.rooms.map(r => ({
            ...r, polygon: r.polygon.map(p => ({ x: p.x, y: p.y })),
        }));
        doors = (floorplanData.floorplan.doors || []).map(d => ({
            ...d, position: { x: d.position.x, y: d.position.y },
        }));
        rooms._boundsX = maxX || 1;
        rooms._boundsY = maxY || 1;
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

    // ── drawing ─────────────────────────────────────────────────────
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (bgImage) {
            const pad = 40, cw = canvas.width - pad * 2, ch = canvas.height - pad * 2;
            const ia = bgImage.width / bgImage.height, ca = cw / ch;
            let dw, dh;
            if (ia > ca) { dw = cw; dh = cw / ia; } else { dh = ch; dw = ch * ia; }
            ctx.globalAlpha = 0.3;
            ctx.drawImage(bgImage, pad, pad, dw, dh);
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

    // ── event handlers ──────────────────────────────────────────────

    canvas.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return; // left click only
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const handle = findHandle(mx, my);
        if (handle) {
            dragState = handle;
            canvas.style.cursor = "grabbing";
        }
    });

    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;

        if (dragState) {
            if (dragState.rotate) {
                const door = doors[dragState.doorIdx];
                const sp = toScreen(door.position);
                const a = Math.atan2(my - sp.y, mx - sp.x);
                door.angle = Math.round(((a * 180 / Math.PI) % 360 + 360) % 360);
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

            if (handle && handle.rotate) canvas.style.cursor = "crosshair";
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

    // Double-click on edge → add point
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
        }
    });

    // Right-click on corner → delete point (min 3 vertices)
    canvas.addEventListener("contextmenu", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const handle = findHandle(mx, my);
        if (handle && handle.roomIdx !== undefined && handle.ptIdx !== undefined) {
            const poly = rooms[handle.roomIdx].polygon;
            if (poly.length > 3) {
                e.preventDefault();
                poly.splice(handle.ptIdx, 1);
                hoveredPoint = null;
                draw();
            }
        }
    });

    // Scroll wheel → rotate doors
    canvas.addEventListener("wheel", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const dIdx = findDoorUnderMouse(mx, my);
        if (dIdx >= 0) {
            e.preventDefault();
            doors[dIdx].angle = ((doors[dIdx].angle || 0) + (e.deltaY > 0 ? 15 : -15) + 360) % 360;
            draw();
        }
    }, { passive: false });

    window.addEventListener("resize", resizeCanvas);

    // ── public API ──────────────────────────────────────────────────

    function resetPolygons() { buildWorkingCopy(); draw(); }

    function getFloorplanData() {
        if (!floorplanData) return null;
        const edited = JSON.parse(JSON.stringify(floorplanData));
        edited.floorplan.rooms = rooms.map(r => ({
            id: r.id, label: r.label, height: r.height, type: r.type,
            polygon: r.polygon.map(p => ({ x: Math.round(p.x * 100) / 100, y: Math.round(p.y * 100) / 100 })),
        }));
        edited.floorplan.doors = doors.map(d => ({
            id: d.id, width: d.width, angle: d.angle, connects: d.connects || [],
            position: { x: Math.round(d.position.x * 100) / 100, y: Math.round(d.position.y * 100) / 100 },
        }));
        return edited;
    }

    function confirmAndRender3D() {
        const data = getFloorplanData();
        if (data) window.viewer.render(data);
    }

    window.editor = { loadImage, resetPolygons, confirmAndRender3D, getFloorplanData };
})();
