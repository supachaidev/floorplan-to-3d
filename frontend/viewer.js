/**
 * 3D Viewer - renders floorplan rooms using Three.js ExtrudeGeometry,
 * with door openings cut into walls.
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

        // Render rooms
        rooms.forEach((room) => {
            const shape = new THREE.Shape();
            const pts = room.polygon;
            if (pts.length < 3) return;

            // Negate Y so 3D orientation matches the 2D editor
            shape.moveTo(pts[0].x - cx, -(pts[0].y - cy));
            for (let i = 1; i < pts.length; i++) {
                shape.lineTo(pts[i].x - cx, -(pts[i].y - cy));
            }
            shape.lineTo(pts[0].x - cx, -(pts[0].y - cy));

            const height = room.height || 3.0;

            const extrudeSettings = {
                steps: 1,
                depth: height,
                bevelEnabled: false,
            };

            const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
            const matConfig = ROOM_MATERIALS[room.type] || ROOM_MATERIALS.other;
            const material = new THREE.MeshPhongMaterial({
                color: matConfig.color,
                opacity: matConfig.opacity,
                transparent: true,
                side: THREE.DoubleSide,
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.rotation.x = -Math.PI / 2;
            mesh.userData.isFloorplan = true;
            mesh.userData.label = room.label;
            scene.add(mesh);

            // Wireframe edges
            const edges = new THREE.EdgesGeometry(geometry);
            const lineMat = new THREE.LineBasicMaterial({
                color: 0xffffff,
                opacity: 0.3,
                transparent: true,
            });
            const wireframe = new THREE.LineSegments(edges, lineMat);
            wireframe.rotation.x = -Math.PI / 2;
            wireframe.userData.isFloorplan = true;
            scene.add(wireframe);

            // Room label
            const spriteMat = makeTextSprite(room.label);
            const labelCx = pts.reduce((s, p) => s + p.x, 0) / pts.length - cx;
            const labelCy = -(pts.reduce((s, p) => s + p.y, 0) / pts.length - cy);
            spriteMat.position.set(labelCx, height + 0.5, labelCy);
            spriteMat.userData.isFloorplan = true;
            scene.add(spriteMat);
        });

        // Render doors as 3D objects
        // The angle from detection is the arc midpoint direction in 2D image space
        // (0=right, 90=down, 180=left, 270=up).
        // In 3D world space (after Y-negate): image-X maps to world-X,
        // image-Y maps to world -Z. So convert:
        //   world angle = -(imageAngle) because Y was negated
        doors.forEach((door) => {
            const pos = door.position;
            const doorWidth = door.width || 0.9;
            const doorHeight = 2.1; // Standard door height

            // Hinge position in world space (same transform as rooms)
            const hx = pos.x - cx;
            const hz = -(pos.y - cy);

            // Convert 2D image angle to 3D XZ-plane angle
            // Image: 0=right(+X), 90=down(+Y). World: X=X, Z=-Y
            // So world angle from +X axis = -imageAngle
            const imgAngleRad = (door.angle || 0) * Math.PI / 180;
            const worldAngle = -imgAngleRad;

            // The door panel (closed position) extends from hinge along the arc edge.
            // A quarter-circle arc spans from startAngle to startAngle+90.
            // The midpoint angle = startAngle + 45. So the closed-door edge is at
            // midAngle - 45° and the open-door edge is at midAngle + 45°.
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

            // Door panel as a thin box, pivoted from one end
            // Place it at the closed position (one edge of the arc)
            const panelGeo = new THREE.BoxGeometry(doorWidth, doorHeight, 0.05);
            // Shift geometry so pivot is at one end (hinge side)
            panelGeo.translate(doorWidth / 2, 0, 0);
            const panel = new THREE.Mesh(panelGeo, doorMat);
            panel.position.set(hx, doorHeight / 2, hz);
            panel.rotation.y = -closedAngle; // rotate around Y to align with closed edge
            panel.userData.isFloorplan = true;
            scene.add(panel);

            // Door frame posts
            const postGeo = new THREE.BoxGeometry(0.08, doorHeight + 0.1, 0.15);

            // Hinge post
            const postHinge = new THREE.Mesh(postGeo, frameMat);
            postHinge.position.set(hx, doorHeight / 2, hz);
            postHinge.userData.isFloorplan = true;
            scene.add(postHinge);

            // Latch-side post (at the end of closed-door position)
            const latchX = hx + doorWidth * Math.cos(closedAngle);
            const latchZ = hz + doorWidth * Math.sin(closedAngle);
            const postLatch = new THREE.Mesh(postGeo, frameMat);
            postLatch.position.set(latchX, doorHeight / 2, latchZ);
            postLatch.userData.isFloorplan = true;
            scene.add(postLatch);

            // Top beam connecting the two posts
            const beamLen = doorWidth + 0.08;
            const beamGeo = new THREE.BoxGeometry(beamLen, 0.1, 0.15);
            beamGeo.translate(beamLen / 2 - 0.04, 0, 0);
            const beam = new THREE.Mesh(beamGeo, frameMat);
            beam.position.set(hx, doorHeight + 0.05, hz);
            beam.rotation.y = -closedAngle;
            beam.userData.isFloorplan = true;
            scene.add(beam);

            // Door swing arc on the floor
            // EllipseCurve works in 2D (X,Y), then we rotate to XZ plane
            const arcStart = -openAngle;  // negate because EllipseCurve Y is up
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

            // Radial lines from hinge to arc endpoints on the floor
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
        // Merge all room polygons into a single floor shape
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
            mesh.position.y = 0.01; // Slightly above grid
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
