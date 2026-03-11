/**
 * 3D Viewer - renders floorplan as wall segments with door openings,
 * using Three.js.
 */
(function () {
    const container = document.getElementById("viewer-3d");
    let scene, camera, renderer, controls;
    let initialized = false;

    const ROOM_MATERIALS = {
        bedroom: { color: 0x6495ed, opacity: 0.7 },
        bathroom: { color: 0x00ced1, opacity: 0.7 },
        kitchen: { color: 0xffa500, opacity: 0.7 },
        living: { color: 0x90ee90, opacity: 0.7 },
        dining: { color: 0xffc864, opacity: 0.7 },
        hallway: { color: 0xb4b4c8, opacity: 0.6 },
        closet: { color: 0xa08cb4, opacity: 0.6 },
        balcony: { color: 0x78c8c8, opacity: 0.6 },
        other: { color: 0xcccccc, opacity: 0.6 },
    };

    const WALL_THICKNESS = 0.15;
    const DOOR_HEIGHT = 2.1;

    function init() {
        if (initialized) return;
        initialized = true;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111);

        const w = container.clientWidth;
        const h = container.clientHeight;
        camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
        camera.position.set(15, 20, 15);
        camera.lookAt(0, 0, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(w, h);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.minDistance = 3;
        controls.maxDistance = 100;

        // Lights
        const ambient = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambient);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(10, 20, 10);
        dirLight.castShadow = true;
        scene.add(dirLight);

        const hemiLight = new THREE.HemisphereLight(0xddeeff, 0x333333, 0.3);
        scene.add(hemiLight);

        // Grid
        const grid = new THREE.GridHelper(30, 30, 0x333333, 0x222222);
        scene.add(grid);

        window.addEventListener("resize", onResize);
        animate();
    }

    function onResize() {
        if (!renderer) return;
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    }

    function animate() {
        requestAnimationFrame(animate);
        if (controls) controls.update();
        if (renderer && scene && camera) renderer.render(scene, camera);
    }

    function clearScene() {
        if (!scene) return;
        const toRemove = [];
        scene.traverse((obj) => {
            if (obj.userData.isFloorplan) toRemove.push(obj);
        });
        toRemove.forEach((obj) => {
            obj.geometry?.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) {
                    obj.material.forEach((m) => {
                        m.map?.dispose();
                        m.dispose();
                    });
                } else {
                    obj.material.map?.dispose();
                    obj.material.dispose();
                }
            }
            scene.remove(obj);
        });
    }

    /**
     * Project a door onto a wall segment and return the parametric
     * t value (0..1) along the wall, or null if the door is not
     * close enough. Uses the hinge point (on the wall) if available,
     * falling back to the centroid position.
     */
    function doorOnWall(ax, ay, bx, by, door) {
        const dx = bx - ax, dy = by - ay;
        const len2 = dx * dx + dy * dy;
        if (len2 < 1e-8) return null;

        const ref = door.position;
        const px = ref.x - ax;
        const py = ref.y - ay;

        // Parametric projection onto line segment
        let t = (px * dx + py * dy) / len2;
        if (t < 0 || t > 1) return null;

        // Perpendicular distance from reference point to wall line
        const projX = ax + t * dx;
        const projY = ay + t * dy;
        const dist = Math.sqrt((ref.x - projX) ** 2 + (ref.y - projY) ** 2);

        // Allow doors within a reasonable distance from the wall
        const wallLen = Math.sqrt(len2);
        const threshold = Math.max(WALL_THICKNESS * 4, wallLen * 0.08);
        if (dist > threshold) return null;

        return { t, width: door.width || 0.9 };
    }

    /**
     * Build a wall quad (two triangles) between two 3D points at
     * given Y range, offset by the wall normal for thickness.
     * Returns a THREE.Mesh.
     */
    function makeWallMesh(x0, z0, x1, z1, yBottom, yTop, nx, nz, material) {
        // Four corners of the wall face (outer side)
        const o0 = [x0 + nx, yBottom, z0 + nz];
        const o1 = [x1 + nx, yBottom, z1 + nz];
        const o2 = [x1 + nx, yTop, z1 + nz];
        const o3 = [x0 + nx, yTop, z0 + nz];
        // Four corners (inner side)
        const i0 = [x0 - nx, yBottom, z0 - nz];
        const i1 = [x1 - nx, yBottom, z1 - nz];
        const i2 = [x1 - nx, yTop, z1 - nz];
        const i3 = [x0 - nx, yTop, z0 - nz];

        // 6 faces: outer, inner, top, bottom, left cap, right cap
        // prettier-ignore
        const verts = new Float32Array([
            // Outer face
            ...o0, ...o1, ...o2,  ...o0, ...o2, ...o3,
            // Inner face
            ...i1, ...i0, ...i3,  ...i1, ...i3, ...i2,
            // Top face
            ...o3, ...o2, ...i2,  ...o3, ...i2, ...i3,
            // Bottom face
            ...o0, ...i1, ...o1,  ...o0, ...i0, ...i1,
            // Left cap
            ...o0, ...o3, ...i3,  ...o0, ...i3, ...i0,
            // Right cap
            ...o1, ...i2, ...o2,  ...o1, ...i1, ...i2,
        ]);

        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
        geo.computeVertexNormals();

        const mesh = new THREE.Mesh(geo, material);
        mesh.userData.isFloorplan = true;
        return mesh;
    }

    function render(floorplanData) {
        init();
        clearScene();

        const rooms = floorplanData.floorplan.rooms;
        const doors = floorplanData.floorplan.doors || [];
        if (!rooms.length) return;

        // Find centroid of all rooms for centering
        let cx = 0, cy = 0, count = 0;
        for (const room of rooms) {
            for (const pt of room.polygon) {
                cx += pt.x;
                cy += pt.y;
                count++;
            }
        }
        cx /= count;
        cy /= count;

        // Render each room as wall segments with door openings
        rooms.forEach((room) => {
            const pts = room.polygon;
            if (pts.length < 3) return;

            const height = room.height || 3.0;
            const matConfig = ROOM_MATERIALS[room.type] || ROOM_MATERIALS.other;
            const wallMat = new THREE.MeshPhongMaterial({
                color: matConfig.color,
                opacity: matConfig.opacity,
                transparent: true,
                side: THREE.DoubleSide,
            });

            // For each edge of the room polygon, build a wall segment
            for (let i = 0; i < pts.length; i++) {
                const a = pts[i];
                const b = pts[(i + 1) % pts.length];

                const ax = a.x - cx, az = -(a.y - cy);
                const bx = b.x - cx, bz = -(b.y - cy);

                const edgeDx = bx - ax, edgeDz = bz - az;
                const wallLen = Math.sqrt(edgeDx * edgeDx + edgeDz * edgeDz);
                if (wallLen < 0.01) continue;

                // Outward normal for thickness
                const nx = -edgeDz / wallLen * (WALL_THICKNESS / 2);
                const nz = edgeDx / wallLen * (WALL_THICKNESS / 2);

                // Find doors on this wall segment
                const wallDoorHits = [];
                for (const door of doors) {
                    const hit = doorOnWall(a.x, a.y, b.x, b.y, door);
                    if (hit) wallDoorHits.push(hit);
                }

                if (wallDoorHits.length === 0) {
                    // Solid wall, no doors
                    scene.add(makeWallMesh(
                        ax, az, bx, bz, 0, height, nx, nz, wallMat
                    ));
                } else {
                    // Sort doors by position along wall
                    wallDoorHits.sort((a, b) => a.t - b.t);

                    // Split wall into solid sections and door openings
                    let prevT = 0;
                    for (const dh of wallDoorHits) {
                        const halfW = (dh.width / 2) / wallLen;
                        const doorStart = Math.max(0, dh.t - halfW);
                        const doorEnd = Math.min(1, dh.t + halfW);

                        // Solid wall before this door
                        if (doorStart > prevT + 0.001) {
                            const sx = ax + edgeDx * prevT;
                            const sz = az + edgeDz * prevT;
                            const ex = ax + edgeDx * doorStart;
                            const ez = az + edgeDz * doorStart;
                            scene.add(makeWallMesh(
                                sx, sz, ex, ez, 0, height, nx, nz, wallMat
                            ));
                        }

                        // Wall above door opening (lintel)
                        if (DOOR_HEIGHT < height) {
                            const sx = ax + edgeDx * doorStart;
                            const sz = az + edgeDz * doorStart;
                            const ex = ax + edgeDx * doorEnd;
                            const ez = az + edgeDz * doorEnd;
                            scene.add(makeWallMesh(
                                sx, sz, ex, ez, DOOR_HEIGHT, height, nx, nz,
                                wallMat
                            ));
                        }

                        prevT = doorEnd;
                    }

                    // Solid wall after last door
                    if (prevT < 1 - 0.001) {
                        const sx = ax + edgeDx * prevT;
                        const sz = az + edgeDz * prevT;
                        scene.add(makeWallMesh(
                            sx, sz, bx, bz, 0, height, nx, nz, wallMat
                        ));
                    }
                }
            }

            // Room label
            const spriteMat = makeTextSprite(room.label);
            const labelCx = pts.reduce((s, p) => s + p.x, 0) / pts.length - cx;
            const labelCy = -(pts.reduce((s, p) => s + p.y, 0) / pts.length - cy);
            spriteMat.position.set(labelCx, height + 0.5, labelCy);
            spriteMat.userData.isFloorplan = true;
            scene.add(spriteMat);
        });

        // Render doors as 3D objects
        doors.forEach((door) => {
            const pos = door.position;
            const doorWidth = door.width || 0.9;
            const doorHeight = DOOR_HEIGHT;

            // Door hinge in world space (same transform as rooms)
            const hx = pos.x - cx;
            const hz = -(pos.y - cy);

            // Convert 2D image angle to 3D XZ-plane angle.
            // The detector angle is atan2 from hinge toward arc center in image
            // space (Y-down). In world space Y is negated, but the panel rotation
            // around Y already accounts for the flip, so use the angle directly.
            const imgAngleRad = (door.angle || 0) * Math.PI / 180;
            const worldAngle = imgAngleRad;

            const closedAngle = worldAngle - Math.PI / 4;
            const openAngle = worldAngle + Math.PI / 4;

            // Door frame material
            const frameMat = new THREE.MeshPhongMaterial({
                color: 0x8B4513,
                opacity: 0.9,
                transparent: true,
            });

            // Door panel material
            const doorMat = new THREE.MeshPhongMaterial({
                color: 0xDEB887,
                opacity: 0.85,
                transparent: true,
                side: THREE.DoubleSide,
            });

            // Door panel (thin box pivoted from hinge end)
            const panelGeo = new THREE.BoxGeometry(doorWidth, doorHeight, 0.05);
            panelGeo.translate(doorWidth / 2, 0, 0);
            const panel = new THREE.Mesh(panelGeo, doorMat);
            panel.position.set(hx, doorHeight / 2, hz);
            panel.rotation.y = -closedAngle;
            panel.userData.isFloorplan = true;
            scene.add(panel);

            // Door frame posts
            const postGeo = new THREE.BoxGeometry(0.08, doorHeight + 0.1, 0.15);

            const postHinge = new THREE.Mesh(postGeo, frameMat);
            postHinge.position.set(hx, doorHeight / 2, hz);
            postHinge.userData.isFloorplan = true;
            scene.add(postHinge);

            const latchX = hx + doorWidth * Math.cos(closedAngle);
            const latchZ = hz + doorWidth * Math.sin(closedAngle);
            const postLatch = new THREE.Mesh(postGeo, frameMat);
            postLatch.position.set(latchX, doorHeight / 2, latchZ);
            postLatch.userData.isFloorplan = true;
            scene.add(postLatch);

            // Top beam
            const beamLen = doorWidth + 0.08;
            const beamGeo = new THREE.BoxGeometry(beamLen, 0.1, 0.15);
            beamGeo.translate(beamLen / 2 - 0.04, 0, 0);
            const beam = new THREE.Mesh(beamGeo, frameMat);
            beam.position.set(hx, doorHeight + 0.05, hz);
            beam.rotation.y = -closedAngle;
            beam.userData.isFloorplan = true;
            scene.add(beam);

            // Door swing arc on floor
            const arcStart = -openAngle;
            const arcEnd = -closedAngle;
            const arcCurve = new THREE.EllipseCurve(
                0, 0,
                doorWidth, doorWidth,
                Math.min(arcStart, arcEnd),
                Math.max(arcStart, arcEnd),
                false, 0,
            );
            const arcPoints = arcCurve.getPoints(24);
            const arcGeo = new THREE.BufferGeometry().setFromPoints(arcPoints);
            const arcLineMat = new THREE.LineBasicMaterial({
                color: 0xFF6B6B,
                opacity: 0.6,
                transparent: true,
            });
            const arcLine = new THREE.Line(arcGeo, arcLineMat);
            arcLine.rotation.x = -Math.PI / 2;
            arcLine.position.set(hx, 0.02, hz);
            arcLine.userData.isFloorplan = true;
            scene.add(arcLine);

            // Radial lines
            const radiiGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, 0),
                new THREE.Vector3(
                    doorWidth * Math.cos(closedAngle),
                    0,
                    doorWidth * Math.sin(closedAngle),
                ),
                new THREE.Vector3(0, 0, 0),
                new THREE.Vector3(
                    doorWidth * Math.cos(openAngle),
                    0,
                    doorWidth * Math.sin(openAngle),
                ),
            ]);
            const radiiLine = new THREE.LineSegments(radiiGeo, new THREE.LineBasicMaterial({
                color: 0xFF6B6B,
                opacity: 0.35,
                transparent: true,
            }));
            radiiLine.position.set(hx, 0.02, hz);
            radiiLine.userData.isFloorplan = true;
            scene.add(radiiLine);
        });

        // Render floor plane
        renderFloor(rooms, cx, cy);

        // Fit camera
        const bbox = new THREE.Box3();
        scene.traverse((obj) => {
            if (obj.userData.isFloorplan && (obj.isMesh || obj.isLine)) {
                bbox.expandByObject(obj);
            }
        });
        if (!bbox.isEmpty()) {
            const center = new THREE.Vector3();
            bbox.getCenter(center);
            const size = bbox.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            camera.position.set(
                center.x + maxDim,
                maxDim * 1.2,
                center.z + maxDim,
            );
            controls.target.copy(center);
            controls.update();
        }
    }

    function renderFloor(rooms, cx, cy) {
        rooms.forEach((room) => {
            const pts = room.polygon;
            if (pts.length < 3) return;

            const shape = new THREE.Shape();
            shape.moveTo(pts[0].x - cx, -(pts[0].y - cy));
            for (let i = 1; i < pts.length; i++) {
                shape.lineTo(pts[i].x - cx, -(pts[i].y - cy));
            }
            shape.lineTo(pts[0].x - cx, -(pts[0].y - cy));

            const geo = new THREE.ShapeGeometry(shape);
            const mat = new THREE.MeshPhongMaterial({
                color: 0x444444,
                opacity: 0.3,
                transparent: true,
                side: THREE.DoubleSide,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.rotation.x = -Math.PI / 2;
            mesh.position.y = 0.01;
            mesh.userData.isFloorplan = true;
            scene.add(mesh);
        });
    }

    function makeTextSprite(text, w, h, fontSize, color) {
        w = w || 256;
        h = h || 64;
        fontSize = fontSize || 28;
        color = color || "#ffffff";

        const canvas2 = document.createElement("canvas");
        canvas2.width = w;
        canvas2.height = h;
        const ctx2 = canvas2.getContext("2d");
        ctx2.fillStyle = "rgba(0,0,0,0.6)";
        if (ctx2.roundRect) {
            ctx2.roundRect(0, 0, w, h, 8);
        } else {
            ctx2.rect(0, 0, w, h);
        }
        ctx2.fill();
        ctx2.fillStyle = color;
        ctx2.font = `bold ${fontSize}px -apple-system, sans-serif`;
        ctx2.textAlign = "center";
        ctx2.textBaseline = "middle";
        ctx2.fillText(text, w / 2, h / 2);

        const texture = new THREE.CanvasTexture(canvas2);
        const spriteMat = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
        });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.scale.set(3, 0.75, 1);
        return sprite;
    }

    // Public API
    window.viewer = { render };
})();
