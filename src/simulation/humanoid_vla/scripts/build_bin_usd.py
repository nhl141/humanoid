"""Build assets/props/bin.usd and the humanoid_vla env USD that uses it.

bin.usd is an open-top box, a little smaller than box.usd's 25.4 cm tray but much
deeper, so a dropped cube stays in. It is built from five analytic Cube colliders
(floor + four walls) rather than one mesh: a convex-hull collider -- what box.usd uses
-- fills the hollow, and the cube would rest on a lid instead of dropping in.

box.usd itself is left alone: the push_block RL task spawns it too.

The env USD is 09-19_vla_env_v1.usd with /World/Environment/box re-pointed at bin.usd
(same prim name, so vla_keyboard_teleop's RigidObjectCfg path is unchanged). v1 is
kept as-is.

    ~/anaconda3/envs/env_isaaclab/bin/python scripts/build_bin_usd.py

The dimensions here must match BIN_* in vla_keyboard_teleop.py.
"""

from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

_VLA_DIR = Path(__file__).resolve().parents[1]
_REPO = _VLA_DIR.parents[2]
BIN_USD = _REPO / "assets" / "props" / "bin.usd"
ENV_V1 = _VLA_DIR / "assets" / "09-19_vla_env_v1.usd"
ENV_V2 = _VLA_DIR / "assets" / "09-27_vla_env_v2.usd"

OUTER = 0.20            # outer width and depth, m (box.usd tray is 0.254)
HEIGHT = 0.07           # outer height, m (box.usd tray is 0.032)
WALL = 0.008            # wall and floor thickness, m
MASS = 0.3              # kg
COLOR = (0.62, 0.81, 0.93)   # box.usd's light blue

# Bin geometry centre in world XY -- where box.usd's tray centre sat in v1 -- resting on
# the table top (TABLE_TOP_Z = 0.7507) with 1 mm clearance.
POS = (0.5007, -0.2886, 0.7517)


def build_bin():
    stage = Usd.Stage.CreateNew(str(BIN_USD))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/bin").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(MASS)

    mat = UsdShade.Material.Define(stage, "/bin/Looks/Bin")
    shader = UsdShade.Shader.Define(stage, "/bin/Looks/Bin/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLOR))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Prim origin = bottom centre of the bin. (centre xyz, size xyz) per slab.
    h, o, t = HEIGHT, OUTER, WALL
    slabs = {
        "floor": ((0, 0, t / 2), (o, o, t)),
        "wall_px": ((o / 2 - t / 2, 0, h / 2), (t, o, h)),
        "wall_nx": ((-o / 2 + t / 2, 0, h / 2), (t, o, h)),
        "wall_py": ((0, o / 2 - t / 2, h / 2), (o - 2 * t, t, h)),
        "wall_ny": ((0, -o / 2 + t / 2, h / 2), (o - 2 * t, t, h)),
    }
    for name, (centre, size) in slabs.items():
        cube = UsdGeom.Cube.Define(stage, f"/bin/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*COLOR)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    print(f"[build_bin_usd] wrote {BIN_USD}")


def build_env():
    Sdf.Layer.FindOrOpen(str(ENV_V1)).Export(str(ENV_V2))
    stage = Usd.Stage.Open(str(ENV_V2))
    path = "/World/Environment/box"
    stage.RemovePrim(path)          # drops box.usd's payload and every over under it
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    # Relative to humanoid_vla/assets/, like v1's box.usd payload.
    prim.GetPayloads().AddPayload("../../../../assets/props/bin.usd")
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(POS[0], POS[1], POS[2]))
    xf.AddOrientOp().Set(Gf.Quatf(1.0))
    stage.GetRootLayer().Save()
    print(f"[build_bin_usd] wrote {ENV_V2}")


build_bin()
build_env()
app.close()
