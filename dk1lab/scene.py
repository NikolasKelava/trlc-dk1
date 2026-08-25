"""Build the bimanual DK1 MuJoCo scene out of this repo's own URDF.

``STUDY.md`` asks for a simulator on one condition: it must exercise **the same
pipeline the arms use** and must not favour either policy. That rules out
ManiSkill's BimanualYAM, which is MolmoAct2's own training embodiment and reaches
the policy over a separate HTTP path, and it rules out ``gym-aloha``, which is a
different robot. What is left is this cell's own URDF, which is already in the
repo and which ``trlc_dk1_control/gravity_comp.py`` already loads into MuJoCo —
with the meshes stripped and dynamics only, because inverse dynamics does not
need geometry. A scene you can look at does.

**The scene is generated, not checked in.** ``urdf/`` is upstream's and is not
edited; this module reads it, converts one arm through MuJoCo's own URDF parser,
and composes two of them. Regenerating is a second of work, so there is no
second copy of the robot's geometry to keep in step with the first.

The composition, in order:

1. **One arm to MJCF.** MuJoCo's URDF importer is given a ``<mujoco><compiler>``
   block naming the mesh directory — it strips paths from mesh filenames, which
   is exactly why ``gravity_comp`` replaces them with spheres — and the ``.glb``
   visual meshes are dropped, because MuJoCo reads STL and the collision meshes
   are STL. ``mj_saveLastXML`` then hands back MJCF.
2. **Two arms, prefixed.** Every name in the arm's bodies is prefixed ``left_``
   / ``right_`` and every reference to one is rewritten, so the two copies can
   share one ``<asset>`` block and one set of meshes.
3. **The cell around them.** A table, a light, and the three cameras this cell
   has: ``top`` overhead and one on each wrist at the pose the URDF's own
   ``camera`` link declares.
4. **What the task needs.** A die and a bowl, at the positions ``STUDY.md``'s
   one task names. The bowl is four walls and a base rather than a mesh, which
   is enough to tell "in the bowl" from "not in the bowl".
5. **Actuators.** One position actuator per joint, at the impedance gains the
   real arms run — ``arm_kp = [100, 100, 100, 20, 20, 10]`` from
   ``trlc_dk1_control/config.py`` — so a joint that lags in the sim lags for a
   reason the cell shares. The two fingers of a gripper are driven together.

**Nothing here is a claim about the cell's real geometry.** The arm spacing and
the camera poses are the *training rig's*, taken from ``sim_eval``: arms at
y = ±0.24 m facing +X, an overhead view and two wrist views. The sim exists to
confirm a policy drives the pipeline, not to predict what it will do on the
bench — ``STUDY.md`` is explicit that it produces no episodes and no scores.
"""

from __future__ import annotations

import copy
import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .layout import ARMS

logger = logging.getLogger(__name__)

#: This repo's follower URDF. Upstream's file, read and never written.
DEFAULT_URDF = Path(__file__).resolve().parent.parent / "urdf" / "follower" / "TRLC-DK1-Follower.urdf"

#: Where each arm's base sits, and which way it faces. The training rig's
#: arrangement, from ``sim_eval/robots/bimanual_yam.py``: both arms facing +X,
#: 0.48 m apart.
ARM_Y: dict[str, float] = {"left": 0.24, "right": -0.24}

#: Height of the table top the objects sit on.
TABLE_Z: float = 0.0

#: How high each arm's base is mounted above the table, and the pedestal that
#: holds it up.
#:
#: **Not cosmetic.** At the zero pose the DK1's elbow folds *behind and below*
#: its own base — ``link3-4`` ends up at x = -0.24 m, z = +0.10 m relative to it —
#: so an arm bolted flat to the table starts the run with eight contact points
#: through the table top, and the base yaw is pinned by them. The first sim
#: rollout would have found a policy that "cannot move joint 1", which is a fact
#: about this scene and not about any policy.
ARM_Z: float = 0.30
PEDESTAL_RADIUS: float = 0.05

#: Contact groups, and why an arm does not collide with itself.
#:
#: MuJoCo lets two geoms touch when ``contype1 & conaffinity2`` or
#: ``contype2 & conaffinity1`` is non-zero. Imported straight from the URDF, the
#: DK1's adjacent links overlap by 1.6-2.5 mm — collision meshes are convex hulls
#: and neighbouring hulls meet at the joint — so at the zero pose each arm starts
#: with four contact points *inside itself*, and the base yaw is pinned by them.
#: A hand-written MJCF would list ``<contact><exclude>`` pairs; giving each arm a
#: bit nothing of its own answers to does the same job in three numbers and does
#: not have to be revisited when the URDF gains a link.
#:
#: left: type 2, affinity 5 (world + right). right: type 4, affinity 3 (world +
#: left). Everything else: type 1, affinity 7 — the world and the task objects
#: collide with both arms and with each other.
ARM_CONTACT: dict[str, tuple[int, int]] = {"left": (2, 5), "right": (4, 3)}
WORLD_CONTACT: tuple[int, int] = (1, 7)

#: Impedance gains the real arms run at (``trlc_dk1_control/config.py``). Used as
#: the position actuators' ``kp`` so a joint that lags here lags for a reason the
#: cell shares.
ARM_KP: tuple[float, ...] = (100.0, 100.0, 100.0, 20.0, 20.0, 10.0)

#: Damping, as a fraction of ``kp``. Not measured on the cell — chosen so the
#: sim arms settle rather than ring, which is the whole of what it is for.
ARM_KD_RATIO: float = 0.1

#: The gripper's travel, in metres of finger displacement, and which end is open.
#:
#: The URDF gives both finger joints the range ``[-0.045, 0.001]`` on opposing
#: axes, so a *larger* value is further apart. This cell's normalised gripper is
#: **0 = open, 1 = closed** (``DK1Robot.command_gripper``), so 0 maps to the top
#: of that range and 1 to the bottom.
GRIPPER_OPEN_M: float = 0.001
GRIPPER_CLOSED_M: float = -0.045
GRIPPER_KP: float = 200.0

#: The six revolute joints, as the URDF names them. The 14-D vector's names come
#: from :mod:`dk1lab.layout`; these are the model's, and :func:`joint_names` is
#: the one place the two are put side by side.
URDF_JOINTS: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
URDF_FINGERS: tuple[str, ...] = ("gripper_left", "gripper_right")

#: The wrist camera's pose on ``link6-7``, straight out of the URDF's own
#: ``camera_joint``. Read here rather than parsed, because the fixed joint is
#: welded away by the time MuJoCo has finished importing.
WRIST_CAMERA_POS: tuple[float, float, float] = (0.0471, 0.0, 0.063314)
WRIST_CAMERA_PITCH: float = 0.2618  # radians, from the URDF's rpy

#: Vertical field of view of each camera, degrees. MuJoCo takes the vertical one
#: and the cell's lenses are quoted horizontally, so these are the conversions at
#: 16:9: the wrists' 105 deg horizontal is 71 deg vertical, and the overhead
#: view's is set to see the whole workspace rather than to match a lens.
#:
#: Note what this is *not*: the crop the `optimized` profile applies lives in the
#: camera (:mod:`dk1lab.crop`) and applies to the real cell's frames. The sim
#: renders the lens's own field of view, which is what `--profile common` sees.
WRIST_CAMERA_FOVY: float = 71.0
TOP_CAMERA_FOVY: float = 58.0

#: Where the task's two objects start. ``STUDY.md`` photographs the real layout
#: to ``study/scene.jpg``; this is the sim's stand-in for it.
DIE_POS: tuple[float, float, float] = (0.45, 0.05, 0.02)
DIE_HALF: float = 0.015
BOWL_POS: tuple[float, float, float] = (0.45, -0.15, 0.0)
BOWL_RADIUS: float = 0.07
BOWL_HEIGHT: float = 0.04


#: The largest frame the scene can render. MuJoCo's offscreen framebuffer is
#: fixed at model-compile time and defaults to 640x480; asking the renderer for
#: more than it raises rather than degrading. 1280x720 is this cell's
#: ``[capture.policy]``, so the sim can render whatever the real cameras deliver.
MAX_RENDER_WIDTH: int = 1280
MAX_RENDER_HEIGHT: int = 720


class SceneError(RuntimeError):
    """Raised when the scene cannot be built, with the reason."""


def joint_names(arm: str) -> list[str]:
    """The MuJoCo joint names of one arm, in ``dk1lab.layout.ARM_KEYS`` order.

    Six revolute joints then the gripper — except that the gripper is two
    prismatic finger joints in the model, so this returns the *left* finger as
    the seventh and the right one is driven with it. Everything that maps the
    14-D vector onto the model goes through here.
    """
    return [f"{arm}_{name}" for name in URDF_JOINTS] + [f"{arm}_{URDF_FINGERS[0]}"]


def actuator_names(arm: str) -> tuple[list[str], list[str]]:
    """One arm's actuators: ``(six joint actuators, two finger actuators)``.

    Split rather than flat because the shapes genuinely differ — this cell has
    **one** gripper channel per arm and the model has **two** finger joints, so
    the seventh element of the 14-D vector drives both of the second list. Keeping
    that asymmetry visible here is better than a seven-long list whose last entry
    silently means two things.
    """
    return (
        [f"{arm}_{name}_act" for name in URDF_JOINTS],
        [f"{arm}_{finger}_act" for finger in URDF_FINGERS],
    )


def gripper_position(normalised: float) -> float:
    """A normalised gripper command (0 = open, 1 = closed) as finger displacement.

    Clamped, because a policy that has never seen this cell can and does ask for
    values outside [0, 1], and a joint target past its limit is a model that
    fights itself rather than an error anyone would notice.
    """
    value = min(1.0, max(0.0, float(normalised)))
    return GRIPPER_OPEN_M + value * (GRIPPER_CLOSED_M - GRIPPER_OPEN_M)


def gripper_normalised(position: float) -> float:
    """Inverse of :func:`gripper_position`, for reporting the measured gripper."""
    span = GRIPPER_CLOSED_M - GRIPPER_OPEN_M
    if span == 0:
        return 0.0
    return min(1.0, max(0.0, (float(position) - GRIPPER_OPEN_M) / span))


# --------------------------------------------------------------------------- #
# One arm
# --------------------------------------------------------------------------- #


def _urdf_for_mujoco(urdf_path: Path) -> str:
    """The URDF with what MuJoCo's importer needs, and without what it cannot read.

    Two edits, both about the importer rather than about the robot:

    * a ``<mujoco><compiler>`` block naming the collision mesh directory, because
      the importer strips the directory off every mesh filename — the same fact
      that made ``gravity_comp`` replace the meshes with spheres;
    * the ``<visual>`` elements removed, because they point at ``.glb`` files and
      MuJoCo reads STL. The collision meshes are the same geometry at lower
      resolution, and they are what the scene is drawn from.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    meshdir = urdf_path.parent / "meshes" / "collision"
    if not meshdir.is_dir():
        raise SceneError(f"{meshdir} does not exist; the URDF's collision meshes are missing")

    block = ET.Element("mujoco")
    ET.SubElement(
        block,
        "compiler",
        {
            "meshdir": str(meshdir),
            "balanceinertia": "true",
            "discardvisual": "false",
            "strippath": "true",
        },
    )
    root.insert(0, block)

    for link in root.iter("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    return ET.tostring(root, encoding="unicode")


def _one_arm_mjcf(urdf_path: Path) -> ET.Element:
    """One arm as MJCF, via MuJoCo's own URDF importer."""
    import tempfile

    import mujoco

    model = mujoco.MjModel.from_xml_string(_urdf_for_mujoco(urdf_path))
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "arm.xml"
        mujoco.mj_saveLastXML(str(out), model)
        return ET.fromstring(out.read_text())


#: Attributes that name something else in the model and therefore have to be
#: rewritten when a body is prefixed. ``mesh`` is deliberately absent: the two
#: arms share one asset block, which is the point of prefixing only the bodies.
_REFERENCES = ("joint", "body", "site", "geom", "class", "childclass", "material")


def _prefix(element: ET.Element, prefix: str, *, rename: set[str]) -> None:
    """Prefix every name under ``element``, and every reference to one.

    ``rename`` is the set of names that belong to the arm — collected in a first
    pass — so that a reference to something shared (a mesh, a texture) is left
    alone while a reference to one of the arm's own joints is rewritten.
    """
    for node in element.iter():
        name = node.get("name")
        if name is not None and name in rename:
            node.set("name", f"{prefix}{name}")
        for attribute in _REFERENCES:
            value = node.get(attribute)
            if value is not None and value in rename:
                node.set(attribute, f"{prefix}{value}")


def _names_in(element: ET.Element) -> set[str]:
    """Every ``name`` declared under ``element``."""
    return {node.get("name") for node in element.iter() if node.get("name")}  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# The scene
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scene:
    """A built scene: the MJCF, and where it came from."""

    xml: str
    urdf: Path

    def write(self, path: Path | str) -> Path:
        """Save the MJCF, for looking at with ``python -m mujoco.viewer``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.xml)
        return path


def build(urdf_path: Path | str = DEFAULT_URDF, *, objects: bool = True) -> Scene:
    """The bimanual DK1 scene, as MJCF.

    Args:
        urdf_path: this repo's follower URDF. Read, never written.
        objects: include the die and the bowl. Off leaves the arms and the
            cameras alone, which is the *kinematic check* ``STUDY.md`` names as
            the acceptable fallback if contact tuning slips.
    """
    urdf_path = Path(urdf_path).expanduser()
    if not urdf_path.is_file():
        raise SceneError(f"{urdf_path} does not exist")

    arm = _one_arm_mjcf(urdf_path)
    asset = arm.find("asset")
    arm_world = arm.find("worldbody")
    if asset is None or arm_world is None:
        raise SceneError("the imported arm has no <asset> or <worldbody>; MuJoCo's output changed")
    own_names = _names_in(arm_world)

    root = ET.Element("mujoco", {"model": "dk1-bimanual"})
    ET.SubElement(root, "compiler", dict(arm.find("compiler").attrib))  # type: ignore[union-attr]
    ET.SubElement(root, "option", {"timestep": "0.002", "integrator": "implicitfast"})
    # MuJoCo's offscreen framebuffer defaults to 640x480 and refuses to render
    # anything larger, so a scene without this cannot produce a frame at this
    # cell's [capture.policy] size of 1280x720 — it raises, at the first tick of
    # a rollout. The timestep above is replaced at connect: see
    # dk1lab.sim.SimRobot.connect for why one tick has to be exactly one period.
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {
        "offwidth": str(MAX_RENDER_WIDTH), "offheight": str(MAX_RENDER_HEIGHT),
    })
    root.append(asset)
    _add_materials(asset)

    world = ET.SubElement(root, "worldbody")
    _add_cell(world)
    for side in ARMS:
        _add_arm(world, arm_world, side, own_names)
    if objects:
        _add_objects(world)
    _add_actuators(root)
    # No <keyframe>: the model's own zero pose already is dk1.toml's [home] —
    # every joint at 0 and both grippers open, which is what was captured on the
    # hardware. Writing the numbers here would be a second copy of [home] that
    # nothing keeps in step with the first.

    return Scene(xml=ET.tostring(root, encoding="unicode"), urdf=urdf_path)


def _add_materials(asset: ET.Element) -> None:
    ET.SubElement(asset, "texture", {
        "name": "grid", "type": "2d", "builtin": "checker", "width": "512", "height": "512",
        "rgb1": ".2 .25 .3", "rgb2": ".25 .3 .35",
    })
    ET.SubElement(asset, "material", {
        "name": "grid", "texture": "grid", "texrepeat": "6 6", "reflectance": ".1",
    })
    # A sky, so that a camera pointing at nothing renders as *something*. A black
    # frame is what a broken renderer produces too, and the two should not look
    # the same while the camera poses are being got right.
    ET.SubElement(asset, "texture", {
        "name": "sky", "type": "skybox", "builtin": "gradient",
        "rgb1": ".35 .42 .5", "rgb2": ".05 .06 .08", "width": "256", "height": "256",
    })


def _contact(attributes: dict[str, str], groups: tuple[int, int]) -> dict[str, str]:
    """``attributes`` plus a contact group. See :data:`ARM_CONTACT`."""
    contype, conaffinity = groups
    return {**attributes, "contype": str(contype), "conaffinity": str(conaffinity)}


def _add_cell(world: ET.Element) -> None:
    """The table, the light, and the overhead camera."""
    ET.SubElement(world, "light", {"pos": "0.4 0 2.0", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(world, "geom", _contact({
        "name": "table", "type": "plane", "size": "1.5 1.5 0.01",
        "pos": f"0 0 {TABLE_Z}", "material": "grid",
    }, WORLD_CONTACT))
    # The overhead view. Looking straight down would give the policy no sense of
    # height at all, so it sits back and tilts, which is what the real one does.
    ET.SubElement(world, "camera", {
        "name": "top", "mode": "fixed", "fovy": f"{TOP_CAMERA_FOVY:g}",
        "pos": f"-0.25 0 {TABLE_Z + ARM_Z + 0.75:g}", "xyaxes": "0 -1 0 0.72 0 0.69",
    })


def _add_arm(world: ET.Element, arm_world: ET.Element, side: str, own_names: set[str]) -> None:
    """One copy of the arm, prefixed and placed, with its wrist camera."""
    ET.SubElement(world, "geom", _contact({
        "name": f"{side}_pedestal", "type": "cylinder",
        "size": f"{PEDESTAL_RADIUS:g} {ARM_Z / 2:g}",
        "pos": f"0 {ARM_Y[side]:g} {TABLE_Z + ARM_Z / 2:g}",
        "rgba": "0.3 0.32 0.35 1",
    }, WORLD_CONTACT))
    holder = ET.SubElement(world, "body", {
        "name": f"{side}_base",
        "pos": f"0 {ARM_Y[side]:g} {TABLE_Z + ARM_Z:g}",
    })
    contype, conaffinity = ARM_CONTACT[side]
    for child in arm_world:
        node = copy.deepcopy(child)
        _prefix(node, f"{side}_", rename=own_names)
        # Every geom of this arm, including the loose base one: an arm answers to
        # the world and to the other arm, and to nothing of its own.
        for geom in ([node] if node.tag == "geom" else []) + list(node.iter("geom")):
            geom.set("contype", str(contype))
            geom.set("conaffinity", str(conaffinity))
        # Gravity compensation, on every link. This is not a convenience: the real
        # cell runs `trlc_dk1_control/gravity_comp.py`, which computes the gravity
        # torques by MuJoCo inverse dynamics and adds them to every command. Without
        # it here the position actuators — at the cell's own modest gains, 20 Nm/rad
        # at the wrist — would droop under load, and a policy commanding absolute
        # joint poses would see a steady-state error the real arms do not have.
        for body in ([node] if node.tag == "body" else []) + list(node.iter("body")):
            body.set("gravcomp", "1")
        holder.append(node)

    wrist = _find_body(holder, f"{side}_link6-7")
    if wrist is None:
        raise SceneError(
            f"no {side}_link6-7 in the imported arm, so the wrist camera cannot be placed"
        )
    # MuJoCo cameras look along their own -z with +y up, and `xyaxes` gives the
    # first two. With x = (0,-1,0) and y = (sin p, 0, cos p) the view direction
    # works out to (cos p, 0, -sin p): straight out along the link's +x, tilted
    # DOWN by the pitch the URDF's own camera_joint declares. The sign matters —
    # tilted up, the wrist sees the sky and renders black.
    pitch = WRIST_CAMERA_PITCH
    ET.SubElement(wrist, "camera", {
        "name": side, "mode": "fixed",
        "fovy": f"{WRIST_CAMERA_FOVY:g}",
        "pos": " ".join(f"{v:g}" for v in WRIST_CAMERA_POS),
        "xyaxes": f"0 -1 0 {math.sin(pitch):g} 0 {math.cos(pitch):g}",
    })


def _find_body(element: ET.Element, name: str) -> ET.Element | None:
    for body in element.iter("body"):
        if body.get("name") == name:
            return body
    return None


def _add_objects(world: ET.Element) -> None:
    """The die and the bowl — ``STUDY.md``'s one task, in the sim."""
    die = ET.SubElement(world, "body", {
        "name": "die", "pos": " ".join(f"{v:g}" for v in DIE_POS),
    })
    ET.SubElement(die, "freejoint", {"name": "die_free"})
    ET.SubElement(die, "geom", _contact({
        "name": "die", "type": "box",
        "size": f"{DIE_HALF:g} {DIE_HALF:g} {DIE_HALF:g}",
        "rgba": "0.9 0.9 0.85 1", "mass": "0.01",
        # The real die is picked up by a rubber-tipped gripper. Slippery contact
        # is the failure this exists to avoid being an artefact of.
        "friction": "1.5 0.02 0.001",
    }, WORLD_CONTACT))

    bowl = ET.SubElement(world, "body", {
        "name": "bowl", "pos": " ".join(f"{v:g}" for v in BOWL_POS),
    })
    ET.SubElement(bowl, "geom", _contact({
        "name": "bowl_base", "type": "cylinder",
        "size": f"{BOWL_RADIUS:g} 0.004", "pos": "0 0 0.004",
        "rgba": "0.7 0.35 0.2 1",
    }, WORLD_CONTACT))
    # Four walls rather than a mesh: enough to tell "in the bowl" from "not in
    # the bowl", which is the whole of what the task's success test needs.
    half = BOWL_RADIUS
    for index, (x, y, sx, sy) in enumerate((
        (half, 0.0, 0.004, half), (-half, 0.0, 0.004, half),
        (0.0, half, half, 0.004), (0.0, -half, half, 0.004),
    )):
        ET.SubElement(bowl, "geom", _contact({
            "name": f"bowl_wall_{index}", "type": "box",
            "size": f"{sx:g} {sy:g} {BOWL_HEIGHT / 2:g}",
            "pos": f"{x:g} {y:g} {BOWL_HEIGHT / 2:g}",
            "rgba": "0.7 0.35 0.2 1",
        }, WORLD_CONTACT))


def _add_actuators(root: ET.Element) -> None:
    """One position actuator per joint, at the cell's own impedance gains."""
    actuators = ET.SubElement(root, "actuator")
    for side in ARMS:
        for index, joint in enumerate(URDF_JOINTS):
            kp = ARM_KP[index]
            ET.SubElement(actuators, "position", {
                "name": f"{side}_{joint}_act", "joint": f"{side}_{joint}",
                "kp": f"{kp:g}", "kv": f"{kp * ARM_KD_RATIO:g}",
            })
        # Both fingers on one command: the cell has one gripper channel per arm,
        # and a model that let the two fingers disagree would not be this robot.
        for finger in URDF_FINGERS:
            ET.SubElement(actuators, "position", {
                "name": f"{side}_{finger}_act", "joint": f"{side}_{finger}",
                "kp": f"{GRIPPER_KP:g}", "kv": f"{GRIPPER_KP * ARM_KD_RATIO:g}",
            })
