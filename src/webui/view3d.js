/* The 3D room view: the scan mesh, the floor overlays, the bins, and dragging them.
 *
 * This is the browser's answer to src/place3d.py, and it shows the same things in the same colours:
 * the scan itself as a backdrop, green free floor, blue push corridor, red bins that are there today,
 * green bins we propose, magenta entrances. Anyone who knows the desktop viewer should not have to
 * learn a second language of colour.
 *
 * WHAT IT DOES NOT DO. It does not decide anything. Dragging is local and instant; the moment you let
 * go, the layout goes to Python and the answer that comes back is the one that counts. The checks in
 * here (on scanned floor, overlapping) are only so the box can turn red under your hand instead of
 * after a round trip.
 *
 * COORDINATES. room.glb is written in the pipeline's gravity-aligned frame, which is right-handed and
 * Y-up -- the same convention glTF uses -- so a bin at plan.json's (x, z) is at three.js's (x, z) and
 * no axis is flipped anywhere in this file. plan.floor_y is the height of the floor plane in that same
 * frame. If the room ever appears on its side or mirrored, the mistake is in glb.py, not here.
 */

// "three" is not a URL -- it is resolved by the <script type="importmap"> in room.html. Both this file
// and the vendored OrbitControls/GLTFLoader ask for the same bare name, so the browser loads three.js
// exactly once; importing it here by relative path instead would give the addons a SECOND copy, and
// then `instanceof` checks across the two fail in ways that are miserable to debug.
import * as THREE from "three";
import { OrbitControls } from "./vendor/controls/OrbitControls.js";
import { GLTFLoader } from "./vendor/loaders/GLTFLoader.js";
import { containsPoint, localProblems, snapYaw } from "./geom.js";

/** Heights of the bin types, metres. BIN_TYPES in annotations.py is (length, height, width) and
 *  plan.bin_types carries all three, so this only has to name a fallback for a type we do not know. */
const FALLBACK_HEIGHT = 1.2;

/** Overlay planes are lifted off the floor by this much so the GPU does not have to choose between
 *  two surfaces at the same depth (which flickers as you orbit -- "z-fighting"). Each layer gets its
 *  own step, so corridor draws over free floor and never argues with it. */
const LIFT = { floor: 0.012, corridor: 0.02, outline: 0.03 };

const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** A CSS colour -> a THREE.Color in the working (linear) space.
 *  setStyle's second argument tells three.js the string is sRGB, which is what CSS colours are; without
 *  it every overlay comes out too bright. */
function themeColor(name) {
  return new THREE.Color().setStyle(cssVar(name), THREE.SRGBColorSpace);
}

export function createView3D(host, ctx) {
  const canvas = document.createElement("canvas");
  canvas.className = "view3d";
  host.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  // Capped at 2: a 3x phone screen would quadruple the pixels for no visible gain on a 120k mesh.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = themeColor("--surface");

  const camera = new THREE.PerspectiveCamera(48, 1, 0.05, 500);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  // Stop just short of horizontal so you cannot orbit under the floor and look at the room from
  // beneath, which is disorienting and shows nothing (the mesh has no underside).
  controls.maxPolarAngle = Math.PI / 2 - 0.04;
  controls.minDistance = 1.2;
  controls.maxDistance = 90;
  controls.screenSpacePanning = false;

  // Sun plus sky. The scan colours are already baked in, so the light is only there to give the
  // geometry shape -- turn it up and the room bleaches out.
  const hemi = new THREE.HemisphereLight(0xffffff, 0x8d8b82, 2.0);
  const sun = new THREE.DirectionalLight(0xffffff, 0.9);
  sun.position.set(0.6, 1, 0.35);
  scene.add(hemi, sun);

  /* Groups, so a layer can be rebuilt or hidden without touching the others. */
  const meshGroup = new THREE.Group();
  const floorGroup = new THREE.Group();
  const binGroup = new THREE.Group();
  const markGroup = new THREE.Group();
  scene.add(meshGroup, floorGroup, binGroup, markGroup);

  const state = {
    binMeshes: [],        // one entry per ctx.bins index: {box, edges, group}
    drag: null,
    hovered: -1,
    routePoints: null,
    disposed: false,
    showScan: true,
    showFloor: true,
  };

  /* ------------------------------------------------------------ the scan */

  const loader = new GLTFLoader();
  let meshLoaded = false;

  function loadScan() {
    const info = ctx.plan.mesh;
    if (!info) {
      ctx.onStatus("Dette skannet har ingen 3D-modell — bruk planvisningen.", "warn");
      return;
    }
    ctx.onProgress(0, `laster 3D-modell (${(info.bytes / 1024 ** 2).toFixed(1)} MB)`);
    loader.load(
      `/api/mesh/${encodeURIComponent(ctx.plan.scan)}`,
      (gltf) => {
        if (state.disposed) return;
        gltf.scene.traverse((node) => {
          if (!node.isMesh) return;
          // The scan is scenery: it must never swallow a click meant for a bin, and it must never
          // cast the bins into shadow.
          node.raycast = () => {};
          node.material.vertexColors = true;
          node.material.side = THREE.DoubleSide;
        });
        meshGroup.add(gltf.scene);
        meshLoaded = true;
        ctx.onProgress(1, "");
        render();
      },
      (event) => {
        if (event.total) ctx.onProgress(event.loaded / event.total, "laster 3D-modell");
      },
      (error) => {
        ctx.onStatus(`Kunne ikke laste 3D-modellen: ${error.message || error}`, "err");
        ctx.onProgress(1, "");
      },
    );
  }

  /* ---------------------------------------------------- floor overlays */

  /** The free-space grids as textures laid over the floor: one for the floor states, one for the push
   *  corridor on top of it.
   *
   *  A mesh per cell would be tens of thousands of objects for a grid that is routinely 200 x 200; a
   *  texture is one draw call and reads identically.
   */
  function buildFloor() {
    floorGroup.clear();
    const masks = ctx.masks;
    if (!masks || !state.showFloor) return;
    const { cols, rows, bits } = masks;

    const free = themeColor("--free"), occupied = themeColor("--occupied");
    const unknown = themeColor("--unknown"), path = themeColor("--path");

    const make = (paint) => {
      const data = new Uint8Array(cols * rows * 4);
      let any = false;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const rgba = paint(masks.data[r * cols + c]);
          if (!rgba) continue;
          any = true;
          const i = (r * cols + c) * 4;
          data[i] = rgba[0] * 255; data[i + 1] = rgba[1] * 255;
          data[i + 2] = rgba[2] * 255; data[i + 3] = rgba[3] * 255;
        }
      }
      if (!any) return null;
      const texture = new THREE.DataTexture(data, cols, rows);
      texture.colorSpace = THREE.SRGBColorSpace;
      // NearestFilter keeps the 5 cm cells as crisp squares. Blurring them would suggest a precision
      // the grid does not have -- a cell is free or it is not.
      texture.magFilter = THREE.NearestFilter;
      texture.minFilter = THREE.LinearMipmapLinearFilter;
      texture.generateMipmaps = true;
      // Stated rather than left to the default, because the row order is what floorQuad's texture
      // coordinates are matched against.
      texture.flipY = false;
      texture.needsUpdate = true;
      return texture;
    };

    const layers = [
      {
        lift: LIFT.floor,
        paint: (v) => {
          if (!(v & bits.floor_observed)) return null;
          if (v & bits.free) return [free.r, free.g, free.b, 0.5];
          if (v & bits.occupied) return [occupied.r, occupied.g, occupied.b, 0.45];
          return [unknown.r, unknown.g, unknown.b, 0.3];
        },
      },
      {
        lift: LIFT.corridor,
        paint: (v) => ((v & bits.corridor) ? [path.r, path.g, path.b, 0.55] : null),
      },
    ];

    for (const layer of layers) {
      const texture = make(layer.paint);
      if (!texture) continue;
      const plane = new THREE.Mesh(
        floorQuad(masks, ctx.plan.floor_y + layer.lift),
        new THREE.MeshBasicMaterial({
          map: texture, transparent: true, depthWrite: false,
          side: THREE.DoubleSide, toneMapped: false,
        }),
      );
      plane.raycast = () => {};
      floorGroup.add(plane);
    }
  }

  /** The quad the floor texture is drawn on, written out in world metres.
   *
   *  Built by hand instead of PlaneGeometry-plus-a-rotation on purpose. A plane is born in XY facing
   *  +Z, so laying it flat means rotating about X -- and the two possible signs differ only in whether
   *  the texture's rows run along +Z or -Z. Get it wrong and the overlay is MIRRORED along Z: still a
   *  plausible-looking room, with free floor drawn where the occupied floor is. Naming the corners and
   *  their texture coordinates makes the mapping something you can read instead of derive.
   *
   *  masks.png row 0 is the grid row at origin_z, and column 0 is the column at origin_x. A texture's
   *  v = 0 is its first row of data, so v runs with +Z and u runs with +X.
   */
  function floorQuad(masks, y) {
    const x0 = masks.origin[0], x1 = x0 + masks.cols * masks.cell;
    const z0 = masks.origin[1], z1 = z0 + masks.rows * masks.cell;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute([
      x0, y, z0,   x1, y, z0,   x1, y, z1,   x0, y, z1,
    ], 3));
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute([
      0, 0,        1, 0,        1, 1,        0, 1,
    ], 2));
    geometry.setAttribute("normal", new THREE.Float32BufferAttribute([
      0, 1, 0,     0, 1, 0,     0, 1, 0,     0, 1, 0,
    ], 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    return geometry;
  }

  /** The room's own footprint rect, and the push route, as lines on the floor. */
  function buildMarks() {
    markGroup.clear();
    const y = ctx.plan.floor_y + LIFT.outline;

    const outline = (ctx.plan.room?.corners || []).map(([x, z]) => new THREE.Vector3(x, y, z));
    if (outline.length) {
      outline.push(outline[0].clone());
      markGroup.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(outline),
        new THREE.LineBasicMaterial({ color: themeColor("--outline"), transparent: true, opacity: 0.75 }),
      ));
    }

    // Entrances: a short post you can see from any angle, plus a disc on the floor. place3d uses a
    // magenta sphere; a post reads better in a perspective view because it is not hidden by a bin.
    const entranceColor = themeColor("--entrance");
    for (const [x, z] of ctx.plan.entrances || []) {
      const post = new THREE.Mesh(
        new THREE.CylinderGeometry(0.055, 0.055, 1.1, 12),
        new THREE.MeshStandardMaterial({ color: entranceColor, roughness: 0.5 }),
      );
      post.position.set(x, ctx.plan.floor_y + 0.55, z);
      post.raycast = () => {};
      const disc = new THREE.Mesh(
        new THREE.CircleGeometry(0.28, 24),
        new THREE.MeshBasicMaterial({ color: entranceColor, transparent: true, opacity: 0.7,
                                      side: THREE.DoubleSide, depthWrite: false, toneMapped: false }),
      );
      disc.rotation.x = -Math.PI / 2;
      disc.position.set(x, ctx.plan.floor_y + LIFT.outline, z);
      disc.raycast = () => {};
      markGroup.add(post, disc);
    }

    // The push route the server last reported: dots along the corridor's skeleton, like the PNGs.
    if (state.routePoints && state.routePoints.length) {
      const positions = new Float32Array(state.routePoints.length * 3);
      state.routePoints.forEach(([x, z], i) => {
        positions[i * 3] = x;
        positions[i * 3 + 1] = ctx.plan.floor_y + LIFT.corridor + 0.004;
        positions[i * 3 + 2] = z;
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const dots = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: themeColor("--path"), size: 0.07, sizeAttenuation: true, toneMapped: false,
      }));
      dots.raycast = () => {};
      markGroup.add(dots);
    }
  }

  /* ---------------------------------------------------------------- bins */

  function binHeight(bin) {
    const spec = ctx.plan.bin_types?.[bin.type];
    return spec?.height_m || FALLBACK_HEIGHT;
  }

  /** Rebuild every bin box. Cheap (a handful of boxes) and it keeps one code path for "a bin changed",
   *  "a bin was added" and "a bin was removed" instead of three that can disagree. */
  function buildBins() {
    binGroup.clear();
    state.binMeshes = [];
    ctx.bins.forEach((bin, index) => {
      const height = binHeight(bin);
      const problems = localProblems(ctx.plan, ctx.masks, ctx.bins, index);
      const isNew = bin.source !== "existing";
      const base = themeColor(isNew ? "--bin-new" : "--bin-existing");
      const bad = problems.length > 0;

      const box = new THREE.Mesh(
        new THREE.BoxGeometry(bin.length_m, height, bin.width_m),
        new THREE.MeshStandardMaterial({
          color: bad ? themeColor("--danger") : base,
          transparent: true,
          // A proposal is a suggestion and should read as one: you must be able to see the floor
          // through it. A bin that is already there is solid.
          opacity: isNew ? 0.62 : 0.85,
          roughness: 0.55,
          metalness: 0.0,
          depthWrite: !isNew,
        }),
      );
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(box.geometry),
        new THREE.LineBasicMaterial({
          color: index === ctx.selected() ? themeColor("--ink") : (bad ? themeColor("--danger") : base),
          // A selected bin gets a heavier outline. linewidth is ignored by most WebGL drivers, so
          // selection is shown by COLOUR and by the lift below, never by thickness alone.
        }),
      );
      const group = new THREE.Group();
      group.add(box, edges);
      group.position.set(bin.center[0], ctx.plan.floor_y + height / 2, bin.center[1]);
      group.rotation.y = -(bin.yaw_deg * Math.PI) / 180;
      group.userData.index = index;
      binGroup.add(group);
      state.binMeshes.push({ box, edges, group });
    });
  }

  /* ------------------------------------------------------------- camera */

  /** Frame the room the way place3d opens it: looking down at about 40 degrees, from a corner, with
   *  the whole exported view box in shot. */
  function frameRoom() {
    const view = ctx.plan.view;
    const cx = (view.min[0] + view.max[0]) / 2, cz = (view.min[1] + view.max[1]) / 2;
    const span = Math.max(view.max[0] - view.min[0], view.max[1] - view.min[1]);
    const target = new THREE.Vector3(cx, ctx.plan.floor_y + 0.6, cz);
    // Distance that puts `span` inside the vertical field of view, with a little air around it.
    const distance = (span / 2) / Math.tan((camera.fov * Math.PI) / 360) * 1.28;
    // Along the room's own diagonal, so walls are seen at an angle rather than edge-on.
    const azimuth = ((ctx.plan.room?.angle_deg || 0) + 45) * Math.PI / 180;
    const eye = new THREE.Vector3(
      cx + Math.cos(azimuth) * distance * 0.72,
      ctx.plan.floor_y + distance * 0.66,
      cz + Math.sin(azimuth) * distance * 0.72,
    );
    camera.position.copy(eye);
    controls.target.copy(target);
    controls.update();
  }

  /** Straight down, north up — the 3D equivalent of the plan drawing, for judging a layout. */
  function topDown() {
    const view = ctx.plan.view;
    const cx = (view.min[0] + view.max[0]) / 2, cz = (view.min[1] + view.max[1]) / 2;
    const span = Math.max(view.max[0] - view.min[0], view.max[1] - view.min[1]);
    const distance = (span / 2) / Math.tan((camera.fov * Math.PI) / 360) * 1.15;
    controls.target.set(cx, ctx.plan.floor_y, cz);
    // Not exactly vertical: at a perfectly vertical angle the orbit controls lose which way is up and
    // the view snaps when you next drag.
    camera.position.set(cx + 0.001, ctx.plan.floor_y + distance, cz + 0.001);
    controls.update();
    render();
  }

  /* -------------------------------------------------------- interaction */

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const floorPlane = new THREE.Plane();

  function setPointer(event) {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
  }

  /** Where the pointer meets the floor plane, in world metres. Null when the ray is parallel to the
   *  floor or points at the sky. */
  function pointerOnFloor() {
    floorPlane.set(new THREE.Vector3(0, 1, 0), -ctx.plan.floor_y);
    const hit = new THREE.Vector3();
    return raycaster.ray.intersectPlane(floorPlane, hit) ? hit : null;
  }

  /** Which bin is under the pointer?
   *
   *  Tested against the FOOTPRINT at floor level rather than the 3D box on purpose. Ray-hitting the
   *  box means grabbing a bin by its lid, and then the point you grabbed is a metre above the floor
   *  while the thing you are dragging slides along it -- the box jumps out from under the cursor on the
   *  first move. Picking by footprint makes grab and drag the same plane.
   */
  function binUnderPointer() {
    const hit = pointerOnFloor();
    if (!hit) return -1;
    // Last first: a bin drawn on top of another is the one you meant to grab.
    for (let i = ctx.bins.length - 1; i >= 0; i--) {
      if (containsPoint(ctx.bins[i], hit.x, hit.z)) return i;
    }
    // Nothing on the floor: fall back to the boxes, so a bin seen from the side is still clickable.
    const hits = raycaster.intersectObjects(binGroup.children, true);
    for (const entry of hits) {
      let node = entry.object;
      while (node && node.userData.index === undefined) node = node.parent;
      if (node) return node.userData.index;
    }
    return -1;
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    setPointer(event);
    const index = binUnderPointer();
    ctx.select(index);
    if (index < 0) return;                    // let OrbitControls have the drag: orbit the room
    const hit = pointerOnFloor();
    if (!hit) return;
    state.drag = {
      index,
      dx: ctx.bins[index].center[0] - hit.x,
      dz: ctx.bins[index].center[1] - hit.z,
      moved: false,
    };
    // Orbiting and dragging a bin at the same time would fight over the pointer.
    controls.enabled = false;
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("grabbing");
  });

  canvas.addEventListener("pointermove", (event) => {
    setPointer(event);
    if (!state.drag) {
      const index = binUnderPointer();
      canvas.classList.toggle("overbin", index >= 0);
      if (index !== state.hovered) { state.hovered = index; }
      return;
    }
    const hit = pointerOnFloor();
    if (!hit) return;
    const bin = ctx.bins[state.drag.index];
    bin.center = [hit.x + state.drag.dx, hit.z + state.drag.dz];
    state.drag.moved = true;
    ctx.changed({ live: true });             // redraw panels, do not call the server yet
    buildBins();
    render();
  });

  function endDrag(event) {
    if (!state.drag) return;
    const moved = state.drag.moved;
    state.drag = null;
    controls.enabled = true;
    canvas.classList.remove("grabbing");
    if (event && canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    // The authoritative check runs on drop, never on every pointer move.
    if (moved) ctx.changed({ live: false });
  }

  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  /* --------------------------------------------------------- the frame */

  // Rendering on demand rather than in a constant requestAnimationFrame loop: a still room should not
  // keep a laptop's GPU busy. Damping needs a few frames to settle, so a render schedules the next one
  // while the controls are still moving.
  let pending = false;
  function render() {
    if (state.disposed || pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      if (state.disposed) return;
      const moving = controls.update();
      renderer.render(scene, camera);
      if (moving) render();
    });
  }
  controls.addEventListener("change", render);

  function resize() {
    const width = host.clientWidth, height = host.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    render();
  }

  /* ------------------------------------------------------------- public */

  const api = {
    /** Called once, after plan + masks are loaded. */
    start() {
      buildFloor();
      buildMarks();
      buildBins();
      frameRoom();
      resize();
      loadScan();
      render();
    },
    /** A bin moved / was added / removed / selected. */
    refresh() {
      buildBins();
      render();
    },
    /** The server sent a new push route. */
    setRoute(points) {
      state.routePoints = points;
      buildMarks();
      render();
    },
    /** Rotate the selected bin a quarter turn, snapped to the room axis. */
    rotateSelected() {
      const index = ctx.selected();
      if (index < 0) return;
      const bin = ctx.bins[index];
      bin.yaw_deg = snapYaw(bin.yaw_deg + 90, ctx.plan.room?.angle_deg || 0);
      buildBins();
      render();
    },
    frameRoom() { frameRoom(); render(); },
    topDown,
    toggleScan(on) {
      state.showScan = on === undefined ? !state.showScan : on;
      meshGroup.visible = state.showScan;
      render();
      return state.showScan;
    },
    toggleFloor(on) {
      state.showFloor = on === undefined ? !state.showFloor : on;
      buildFloor();
      render();
      return state.showFloor;
    },
    /** The theme changed (light <-> dark): every colour in here came from CSS, so rebuild. */
    retheme() {
      scene.background = themeColor("--surface");
      buildFloor();
      buildMarks();
      buildBins();
      render();
    },
    hasScan: () => meshLoaded,
    resize,
    dispose() {
      state.disposed = true;
      controls.dispose();
      // Freeing GPU memory is not automatic: geometries and textures outlive the scene graph.
      scene.traverse((node) => {
        node.geometry?.dispose?.();
        const materials = Array.isArray(node.material) ? node.material : [node.material];
        for (const material of materials) {
          if (!material) continue;
          material.map?.dispose?.();
          material.dispose?.();
        }
      });
      renderer.dispose();
      canvas.remove();
    },
  };
  return api;
}
