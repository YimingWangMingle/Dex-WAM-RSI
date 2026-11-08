"""Attributed task guidance, curated from the paper's frozen DexJoCo packages.

This is semantic preparation material, not a controller or an outcome detector.
Numerical corpus analysis is performed separately by the corpus builder. The
phases below must not be assigned to newly selected frames merely by elapsed
time. Recovery suggestions are hypotheses requiring live observation, not
successful recovery trajectories collected by this project.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..runtime.contracts import DUAL as DUAL_CONTRACT, SINGLE as SINGLE_CONTRACT

DATASET_REVISION = '125df5c1019e97503929ef1a0ad8f90373436afa'
PAPER_PACKAGE_REVISION = 'e977b5ad667b4ce753ee947cbb0b95d4fc0fe98c'
PACKAGE_REPOSITORY = 'sensenova/SenseNova-RoboRSI-Packages'
SOURCE_BASE = (f'https://huggingface.co/datasets/{PACKAGE_REPOSITORY}/blob/'
               f'{PAPER_PACKAGE_REVISION}/dexjoco/v1.0.0/rand_full')

# Tuple layout: phase id, name, instruction, observable checks.
# Numerical commands are deliberately absent: geometry must use current RGB,
# matching achieved robot poses, the pinned action contract and first-group
# execution checks. Each task has a different physical completion condition.
_TASKS = {
    'click_mouse': {
        'instruction': 'Move the mouse onto the purple mouse pad and press its left button.',
        'guidance': (
            'Locate the mouse and pad from current wrist RGB with matching achieved state. '
            'Adapt source mouse position/yaw and destination pad separately; the randomized '
            'external view is not a fixed pixel-to-world map. Keep a cursor within a reference '
            'motion and inspect short contact chunks. A blue monitor is only one visual cue: '
            'the mouse must also be on the pad. Do not resume an old home trajectory while touching it.'),
        'phases': [
            ('locate', 'Locate and settle', 'Let the scene settle; identify the pad rim, mouse and left button.',
             ['Check the chosen landmarks in current RGB.', 'Use achieved pose for camera geometry.']),
            ('engage', 'Engage the mouse', 'Approach its sides with the reference hand shape; establish support.',
             ['Mouse lies between opposing fingers and thumb.']),
            ('transport', 'Move onto the pad', 'Slide with stable side contact and adapt the destination to the observed pad.',
             ['Mouse moves with the hand.', 'Mouse body reaches the pad interior.']),
            ('click', 'Position and press', 'Readjust above the left button; align the index before pressing.',
             ['Index contacts the left button, not the wheel.', 'Inspect the monitor for a change.']),
            ('verify', 'Verify and hold', 'Keep the resulting stable pose after checking placement and button feedback.',
             ['Mouse remains on the pad.', 'Monitor change is visually consistent with a click.']),
        ],
        'recovery': [
            ('The hand moves but the mouse stays behind.',
             'Pause transport, relax and realign the side grasp using the current mouse position.'),
            ('The mouse is on the pad but the monitor has not changed.',
             'Lift clear, reidentify the left button, align above it and try short observed presses.'),
            ('The mouse tips or slips away.',
             'Stop the reference, obtain a clear view and reacquire it before continuing.'),
        ],
    },
    'hammer_nail': {
        'instruction': 'Grasp the hammer and drive the nail into the board.',
        'guidance': (
            'Estimate hammer and nail geometry separately from current RGB; table and board height '
            'vary. The wrist site is not the tool tip. Reuse the coupled 16-joint wrap and adapt '
            'pickup to the hammer and the strike to the nail. Original targets can lead achieved '
            'pose substantially: a blocked low target is not a pose to wait for indefinitely. '
            'Keep the hammer held while checking nail depth.'),
        'phases': [
            ('locate', 'Locate hammer and nail', 'Identify the handle, striking face and nail cap; check height and parallax.',
             ['The cap is distinguished from board shadows.', 'Hammer landmarks agree across views.']),
            ('grasp', 'Grasp the handle', 'Approach the shaft with the demonstrated preshape and close the opposing wrap.',
             ['Handle is between fingers and thumb.']),
            ('lift', 'Verify pickup', 'Lift a short distance while retaining the closing command.',
             ['Hammer rises with the hand and clears the table.']),
            ('align', 'Transfer and align', 'Clear the board and place the striking face on a path toward the nail.',
             ['The impact path reaches the nail, with no visible lateral miss.']),
            ('strike', 'Strike and inspect', 'Execute short observed striking chunks; raise the held tool enough to inspect.',
             ['Nail cap descends toward the board surface.', 'Hammer remains in the grasp.']),
        ],
        'recovery': [
            ('The hammer remains on the table after lifting.',
             'Open, retreat upward, update the handle position and retry the wrap.'),
            ('The nail remains protruding after a strike.',
             'Clear the surface, recheck lateral alignment and make a controlled new strike.'),
            ('The hammer slips during transfer or impact.',
             'Stop striking and regrasp; rigid tool-pose estimates are no longer valid.'),
        ],
    },
    'pick_bucket': {
        'instruction': 'Put the boxed food inside the basket, then lift the loaded basket by its handle.',
        'guidance': (
            'The object called a bucket is a shopping basket with a hinged handle. Treat food '
            'insertion and loaded-basket lifting as separate subgoals. Estimate the food and basket '
            'anchors independently, and relocalize the basket after releasing the food. For static '
            'waypoints use measured reference poses plus desired hand shapes; slowed raw action '
            'targets may push below the surface. Lifting the handle alone does not establish a basket lift.'),
        'phases': [
            ('locate', 'Locate box and basket', 'Identify box top and basket rim with current RGB and matching wrist pose.',
             ['Box and basket geometry are distinguished from similarly colored surfaces.']),
            ('grasp_box', 'Grasp and test the box', 'Approach the box sides, close the opposing grasp and lift slightly.',
             ['Box follows the hand and clears the table.']),
            ('place_box', 'Transfer and release food', 'Clear the rim, lower over the interior and release while stationary.',
             ['Entire box footprint lies inside the rim.', 'Released box remains in the basket.']),
            ('hook_handle', 'Acquire and raise the handle', 'Relocalize the near rim; hook beneath the crossbar and follow its hinge arc.',
             ['Crossbar is captured by fingers and thumb.', 'Handle rotates upward.']),
            ('lift_basket', 'Lift the loaded basket', 'Maintain the secured handle grasp and lift smoothly.',
             ['Basket body is clear of the table.', 'Food remains inside.']),
        ],
        'recovery': [
            ('The box is pushed or fails to lift.',
             'Withdraw, reopen and relocalize the displaced box before retrying.'),
            ('Only the handle rises.',
             'Preserve the hook and complete the hinge arc before attempting a full basket lift.'),
            ('The handle slips from the hand.',
             'Return above the current rim, reopen and reacquire the crossbar; do not reuse the old basket pose.'),
        ],
    },
    'pinch_tongs': {
        'instruction': 'Lift the tongs, complete three visible open-close cycles and finish holding them closed.',
        'guidance': (
            'The support fingers retain one handle while the thumb releases and squeezes the other. '
            'Estimate the visible pivot with multiple wrist views when height is uncertain. Adapt '
            'a reference near the observed geometry, preserving its 30 Hz timing. Count visible '
            'opening followed by closure, not command peaks. Tool possession must be verified '
            'throughout; phase landmarks and finger angles do not prove a completed cycle.'),
        'phases': [
            ('locate', 'Locate the pivot', 'Inspect settled wrist views and check that the selected point is the pivot.',
             ['Landmark is not a shadow or jaw tip.', 'Views and achieved poses correspond.']),
            ('seat', 'Seat the handles', 'Approach with the reference preshape and place handles against support fingers and thumb.',
             ['Both intended handle contacts are visible.']),
            ('lift_close', 'Lift and first closure', 'Close while lifting and verify retained possession.',
             ['Tool rises with the hand.', 'A complete closure follows the initial opening.']),
            ('cycle', 'Complete the remaining cycles',
             'After the first closure during lift, complete the remaining two open-close cycles with thumb motion and retained support.',
             ['Count three visible open-close pairs in total, including the first during lift.', 'Tool remains held between cycles.']),
            ('hold', 'Finish closed', 'Maintain the last closed grasp after the visible cycle count is complete.',
             ['Tool is closed and still possessed.']),
        ],
        'recovery': [
            ('An opening or closure is incomplete but the grasp is stable.',
             'Use one additional gentle thumb cycle with arm and support fingers held; observe the whole pair.'),
            ('The tool stays on the table or slips.',
             'Stop counting, re-establish the grasp and only then resume cycle verification.'),
            ('A dropped tool has moved or rotated.',
             'Relocalize its pivot and orientation; do not jump to a later reference phase.'),
        ],
    },
    'fold_glasses': {
        'instruction': 'Fold both glasses temples, place the glasses completely in the open case and release them.',
        'guidance': (
            'Separate supported folding from the later carrying grasp. Re-estimate the second '
            'hinge after the first fold because contact may move the frame. Adapt the glasses '
            'and case separately. The wrist is above the fingertips; wrist position is not '
            'object position. Use live views to verify both folds, full containment and release. '
            'A held historical low command can keep pressing; pause at measured wrist pose with the desired grip.'),
        'phases': [
            ('locate', 'Locate hinges and case', 'Find the extended temples, frame and case interior; resolve front and back.',
             ['Both hinge regions and the case opening are identified.']),
            ('first_fold', 'Fold one temple', 'Support the frame on the table while bracing and curling around one temple.',
             ['Temple rotates behind the frame instead of the whole glasses sliding.']),
            ('second_fold', 'Fold the other temple', 'Disengage, lift and relocalize the remaining extended temple before folding.',
             ['Both temples lie behind the frame.']),
            ('regrasp', 'Secure folded glasses', 'Regrasp a solid rim region with the folded stack and test a small lift.',
             ['Glasses follow the hand without slipping.']),
            ('place', 'Carry and place', 'Clear the case rim, align the full footprint and lower into the interior.',
             ['Entire folded glasses fit inside the rim.']),
            ('release', 'Release and verify', 'Relax after support is established and withdraw upward.',
             ['Glasses remain contained.', 'Hand is detached and both temples remain folded.']),
        ],
        'recovery': [
            ('The whole frame slides instead of folding.',
             'Reopen, lift slightly and re-align the supported brace and opposing finger contact.'),
            ('The glasses do not follow the carrying grasp.',
             'Lower, adjust the grip across a solid frame region and repeat a short test lift.'),
            ('Glasses catch on the case rim or follow the withdrawing hand.',
             'Recenter before placement; if still attached, lower onto support and release further.'),
        ],
    },
    'water_plant': {
        'instruction': 'Lift the spray bottle upright, aim its nozzle into the plant and operate the trigger.',
        'guidance': (
            'Released RGB shows a trigger spray bottle. The middle/ring fingers and thumb '
            'support its body while the index operates the trigger. Locate bottle landmarks '
            'and the plant separately, using achieved poses for calibrated wrist views. Adapt '
            'pickup around the bottle and aim around the plant. Spray appearance is intermittent '
            'and may be occluded; pose tracking alone does not establish watering.'),
        'phases': [
            ('locate', 'Locate bottle and plant', 'Save the pot-rim location and observe bottle body/neck landmarks before contact.',
             ['Landmark correspondence and current views are consistent.']),
            ('grasp', 'Approach and wrap', 'Use the reference preshape around the red body, keeping the index available.',
             ['Thumb opposes supporting fingers around the bottle.']),
            ('lift', 'Lift upright', 'Retain closing targets while lifting and rotating the bottle upright.',
             ['Bottle moves with the hand and does not rotate freely in the grip.']),
            ('aim', 'Aim the nozzle', 'Bring the nozzle toward the foliage using the separately observed plant position.',
             ['Nozzle points into the plant.', 'Bottle remains securely supported.']),
            ('spray', 'Press and inspect', 'Press the trigger with the index while the other fingers continue supporting.',
             ['Check for visible spray toward the leaves.', 'Recheck trigger contact and nozzle aim.']),
        ],
        'recovery': [
            ('The bottle remains behind or rotates loosely during lift.',
             'Lower if needed, relax to the preshape, retreat slightly and re-align the wrap.'),
            ('No spray is visible.',
             'Recheck aim and index contact, then slowly release and press the index again while retaining support.'),
            ('Spray is directed away from the foliage.',
             'Make small observed aim corrections while preserving the supporting finger commands.'),
        ],
    },
    'bimanual_unlock_ipad': {
        'instruction': 'Support the tablet with the left hand and use the right index to enter 1, 2, then 3.',
        'guidance': (
            'Hold the tablet face-up with the left hand; retain its closing command while the '
            'right index taps. The demonstrated acknowledgement is a brief yellow key highlight '
            'and then an orange home screen. Fit the current keypad from noncollinear labeled '
            'keys: 1/2/3 alone do not define a plane. Retract between keys and refit after slip. '
            'No established recovery from a fallen tablet or reset/backspace procedure is supplied.'),
        'phases': [
            ('locate', 'Locate the supported tablet', 'Let it settle, identify a screen corner and use consistent wrist views for depth.',
             ['Selected point is on the screen inside the bezel.']),
            ('grasp', 'Grasp and lift face-up', 'Adapt the left-hand approach; close around the bezel and lift from the stand.',
             ['Tablet follows the left hand.', 'Screen remains face-up and supported.']),
            ('keypad', 'Locate current keys', 'Identify labeled key centers across multiple rows and columns in current right-wrist RGB.',
             ['At least four noncollinear key centers are consistent.', 'Keypad fit matches visible labels.']),
            ('tap', 'Tap 1 then 2 then 3', 'Align above each intended digit, descend incrementally and retract before changing keys.',
             ['Intended digit highlights.', 'Highlight releases after retraction; count acknowledged digits only.']),
            ('verify', 'Verify home screen', 'After the third acknowledged press, retract gently and keep supporting the tablet.',
             ['Orange home screen with icons is visible.', 'Tablet remains stable.']),
        ],
        'recovery': [
            ('Grasp closure misses while the tablet stays on the stand.',
             'Reopen, inspect the current alignment and reseat before lifting.'),
            ('A key fails to acknowledge the press.',
             'Retract, refit the current keypad, realign the index and retry only that unacknowledged digit.'),
            ('Tablet slips or the key plane changes.',
             'Pause tapping and reacquire screen geometry; a left-relative rigid plane estimate may now be wrong.'),
        ],
    },
    'bimanual_hanoi': {
        'instruction': 'Place the medium block on the large block with the right hand, then add the small block with the left.',
        'guidance': (
            'This task is the final two Hanoi moves: medium/red from the middle peg onto '
            'large/blue, then small/yellow from its peg onto that stack. Identify pegs by their '
            'objects, not image-left/right. Keep the large block in place. Clear each source peg '
            'before moving sideways, and visually align the hole before lowering. Keep the '
            'other hand commands unchanged. To pause, use the measured wrist pose while '
            'retaining the last commanded grasp targets.'),
        'phases': [
            ('locate', 'Locate source and destination pegs', 'Identify all blocks and the original large-block peg from current views.',
             ['Destination is the large/blue peg regardless of image direction.']),
            ('medium_grasp', 'Acquire the medium block', 'Straddle and grasp the red block with the right hand, then lift above its peg.',
             ['Block follows the right hand.', 'Underside fully clears the source peg.']),
            ('medium_place', 'Seat medium on large', 'Carry to the destination, align its hole, lower and release after engagement.',
             ['Peg enters the hole.', 'Red block settles on blue.']),
            ('small_grasp', 'Acquire the small block', 'Use the left grasp and lift the yellow block clear while preserving the stack.',
             ['Small block follows the left hand.', 'Its source peg is clear.']),
            ('small_place', 'Seat small on the stack', 'Align the small hole over the destination peg and lower before release.',
             ['Small block seats over medium without destabilizing it.']),
            ('verify', 'Verify the tower', 'Withdraw both hands and allow the stack to settle.',
             ['Blue/red/yellow stack is stable.', 'Other two pegs are empty and hands disengaged.']),
        ],
        'recovery': [
            ('A block does not follow its assigned hand.',
             'Stop transfer; reopen and reposition that hand before another test lift.'),
            ('The hole catches on the destination peg.',
             'Retain the grasp, lift slightly, correct lateral alignment and lower in short observed chunks.'),
            ('The active hand hides the hole or peg.',
             'Use the opposite wrist view before releasing; pixel overlap alone does not establish insertion.'),
        ],
    },
    'bimanual_assembly': {
        'instruction': 'Hold the receiver with the left hand and insert the right-held peg into its central upper hole.',
        'guidance': (
            'Grasp the receiver with the left hand and peg with the right, leaving the free '
            'peg end available for insertion. The four base mounting holes are not the target. '
            'Adapt each pickup to its own object, then express alignment relative to the current '
            'receiver while checking for slip. Keep the left grasp when releasing and withdrawing '
            'the right hand. Geometric reference offsets are proposals, not contact or insertion proofs.'),
        'phases': [
            ('locate', 'Locate the receiver and peg', 'Identify the central upper hole and the peg orientation in current views.',
             ['Target hole is distinguished from base mounting holes.']),
            ('receiver', 'Acquire the receiver', 'Approach with the left preshape, grip its side and test a lift.',
             ['Receiver follows the left hand.', 'Top hole remains accessible.']),
            ('peg', 'Acquire the peg', 'Pinch with the right hand leaving an insertion end free, then lift before rotating.',
             ['Peg follows the right hand.', 'Free end clears surrounding geometry.']),
            ('align', 'Align over the hole', 'Hold the left wrist steady and align shaft with the observed hole axis.',
             ['Shaft angle matches receiver tilt.', 'Tip is centered with visible clearance.']),
            ('insert', 'Insert incrementally', 'Lower in short chunks and use visually justified small corrections.',
             ['Peg enters rather than catching on the rim.', 'Receiver is not twisting away.']),
            ('release', 'Release the peg', 'Release the right grasp after visible support and withdraw along the shaft.',
             ['Peg remains seated after separation.', 'Left still supports the receiver.']),
        ],
        'recovery': [
            ('Either pickup misses.',
             'Retreat and reacquire that object while preserving the other hand grasp.'),
            ('The peg catches or the receiver twists.',
             'Retract along the insertion axis, re-establish alignment and retry short increments.'),
            ('The grasp slips after alignment.',
             'Reinspect relative object geometry before using any wrist-relative target.'),
        ],
    },
    'bimanual_microwave_cook': {
        'instruction': 'Open the microwave, place the food inside, close the door and press START.',
        'guidance': (
            'Use the left hand for the door and right hand for food and START. Preserve the '
            'left door hold while retrying a food grasp. Food pickup geometry and appliance '
            'geometry need separate adaptation. Confirm food support inside and complete '
            'right-hand withdrawal before closing. Preserve released joint-array order; do '
            'not reorder left joints using an XML listing. A printed-looking display is not '
            'proof of activation, and completion cannot be inferred from reaching a frame index.'),
        'phases': [
            ('locate', 'Locate controls and food', 'Identify the door handle, food and START area using current views.',
             ['Food geometry and door sweep are clear.']),
            ('open', 'Engage and open the door', 'Pull and arc the left hand with the hinge while the right remains clear.',
             ['Door moves with the handle grasp.', 'Opening clears food and the entering hand.']),
            ('grasp', 'Acquire the food', 'Close the right grasp around the food and test a small lift.',
             ['Food rises stably with the right hand.']),
            ('place', 'Place and release inside', 'Keep the door open, enter the cavity and release onto support.',
             ['Food is inside, supported and detached.', 'Food clears the sill and doorway.']),
            ('withdraw', 'Withdraw the right hand', 'Move the open hand completely outside before closing.',
             ['All right fingers clear the door sweep.']),
            ('close', 'Close the door', 'Return the left hand along the closing arc and check obstructions.',
             ['Door is flush.', 'Food and fingers are not trapped.']),
            ('start', 'Press START', 'Align the right index with the current START location and press in short chunks.',
             ['Index reaches the intended button.', 'Inspect any visible activation response.']),
        ],
        'recovery': [
            ('Door does not move with the pulling hand.',
             'Re-engage the handle instead of pulling farther with a missed grasp.'),
            ('The food grasp misses.',
             'Keep the door held open; lift the right hand clear, reopen, relocalize food and retry.'),
            ('Door closure stalls.',
             'Withdraw and inspect the sill, food and fingers before another closing attempt.'),
            ('START contact misses.',
             'Reinspect the index pad and button, correct the alignment and advance in short increments.'),
        ],
    },
    'bimanual_photograph': {
        'instruction': 'Hold the camera with the left hand, aim toward the CASIA logo and press the shutter with the right index.',
        'guidance': (
            'Camera pickup, lens aim and right-index shutter contact are separate checks. '
            'Localize camera and logo from current wrist RGB; an initial height prior is unreliable '
            'after tilt or slip. Keep one reference during a grasp and preserve the left closing '
            'targets under load. Aiming corrections should preserve the relative hand arrangement '
            'when appropriate. A completed trajectory or reached wrist pose alone is not evidence of a photograph.'),
        'phases': [
            ('locate', 'Locate camera and target', 'Identify the lens-barrel landmark and logo; obtain another view if cropped.',
             ['Landmarks belong to the intended camera and logo.']),
            ('grasp', 'Approach and close', 'Keep the right hand clear while the left descends and closes around the camera body.',
             ['Camera is between opposing left fingers and thumb.']),
            ('lift', 'Lift and turn', 'Lift from the table and rotate the lens forward while retaining the grasp.',
             ['Camera follows the left hand.', 'Camera remains stable during rotation.']),
            ('aim', 'Aim and settle', 'Orient the lens toward the logo using current visual feedback.',
             ['Lens points toward the target.', 'Relative hand alignment remains suitable for pressing.']),
            ('press', 'Press the shutter', 'Retain left support and bring the right index onto the shutter in observed chunks.',
             ['Index reaches the shutter.', 'Grasp and aim persist during contact.']),
        ],
        'recovery': [
            ('Pickup fails and the camera remains supported on the table.',
             'Stop pressing; retract, reopen, relocalize and establish a new grasp before aiming again.'),
            ('The camera shifts within a held grasp.',
             'Stop pressing, maintain support and closing targets, and reobserve the camera and logo before correcting aim. '
             'If a new grasp is needed, first return the camera to support; do not release it while suspended.'),
            ('The logo is clipped from the current view.',
             'Acquire a clear observation before aiming; retain reference consistency if already holding the camera.'),
            ('The right index misses the shutter.',
             'Retract slightly, inspect the contact view and make small corrections while retaining the left grasp.'),
        ],
    },
}

# SHA-256 values are pinned to the original package manifest, not generated
# from our paraphrases. The build copies the notices and the per-task SOURCES.md.
_SOURCE_HASHES = {
    'bimanual_assembly': ('13c4b5c3c61fc6fcb7475ad0000d7813a0d8fe6320a000f136a9d0453d9e0e79', 'e6bab8b7a1aa8d0fedd969e0741ab4d6dbb6a2790e5f44d8d28384766bca0f76'),
    'bimanual_hanoi': ('fd6ce1e8a8851d371817acceb9810751a0101ff51804c67fac42f49b4c277afd', 'f381b4c77938e9f727d5d674f1a8c5f5fc1570f0f1d7b4fd4158218b5196beef'),
    'bimanual_microwave_cook': ('5e735529d05473a0a59082a1d3b320a556ea7985196acd6772dec4d1e4264847', '1eeff4c0199fe35ac5e6e0012276f794a00a7dff2f179a91488c1ffe811353f3'),
    'bimanual_photograph': ('fe39c89f559d42eea8400f6cc4467e33ba0a207efcfadbce8a6a6bed37be7a70', '648da13bb6bffa071c0d25116628d1d14560b2540f4d8a9844c8715837696133'),
    'bimanual_unlock_ipad': ('89191347ce4964a086338b873dc623d87424206f472f64831e334d9172e1ba58', 'fedfb9056d1fa39687ba4888bf11507689155c0c85a795cc9f7228a415e56e7d'),
    'click_mouse': ('d421f5bbb17b05bea9923c9228e02ac6e8c53399b8a6657d1d51b1e740fa0aa6', 'dffa000cad93b53792ba14b92688500d3406e82af4d63d9c92b4c635885d13a9'),
    'fold_glasses': ('0366f7ca5718cef1561cbb2b31271b8c56d8e13757ac78a3c69b3dbdd15d267a', '17090da49faa853caa35d143c82ba437861037563f4365f470a60c551979c6a1'),
    'hammer_nail': ('27774cb0831d0a60951895d5b0e4bc1aa094cc7ca10c42539509cf3c7c242d56', 'af7df631ad08bc6aa486ba39c5b26d6d68ed02e6224c5974c291a1d2ac06d9dd'),
    'pick_bucket': ('e969bdc174e568a2d11746cb459271ea98013ec72568795cc2013f9ee1ba5dc4', '5ab19b32d1bb8fd246bf990aef554c969c89a733c2793f469ca780f5f3150989'),
    'pinch_tongs': ('74003c88412c2330b08070d86cf9285f1783a04b0fcaa1d191153c6e8bd474f7', 'bba59e5065f187c378c732d85b222906ccbc133c2dc4e9e22e1e85876f516f5c'),
    'water_plant': ('8540108065ff145ff3699c8fcaa74226d581425250423f76ae7f188fac4571f5', '240cf41e2291ed35ccebb9788f6760eed8de2b7b5ba831c90aeaa4a065bcad8b'),
}

TASKS = tuple(sorted(_TASKS))

_NOTICE_HASHES = {
    'LICENSE': '1e6adcd7f84c0887ab1de92e571d7861ce69504967380dbdeec2d60993e63bb0',
    'LICENSE-MIT': '101a2b0afd49b0373fc4283afcb1e90d4a6d51aac4d335b954bed24e52222661',
    'THIRD_PARTY_NOTICES.md': '3275ed9b9001c9fc1447b0388779f6d1cee41f73b9a242b700723d12bf8ab3ca',
    'third_party/DexJoCo-Datasets-Raw-CARD.md': 'd95ff21c8d3e578aa5e993e7d641b2634a19b7856fb6c51bb15ef76bafd3eda5',
    'third_party/DexJoCo-LICENSE': '939691b484c5ffb0681cbbf9bbf2184501b593af218abef5236cd7c9cc360302',
    'third_party/franka_emika_panda-LICENSE': 'a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9',
    'third_party/GeoRT-LICENSE': 'b8e9691a7ef5ba2e1d85cc17c20ce9b17b786068092d3698878830f396c3b151',
    'third_party/wonik_allegro-LICENSE': '09c187439ecece8b00330e71eb988d3d30f29373c580dc31ab54200d0aa4deb4',
}


def copy_attribution(task: str, output_dir: Path) -> None:
    """Copy the pinned component notices and source record next to the task knowledge."""
    if task not in _TASKS:
        raise ValueError(f'unknown DexJoCo task: {task}')
    assets = Path(__file__).with_name('attribution')
    source_record = f'sources/{task}/SOURCES.md'
    expected = {**_NOTICE_HASHES, source_record: _SOURCE_HASHES[task][1]}
    for name, digest in expected.items():
        data = (assets / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'attribution file differs from pinned revision: {name}')
        dest = Path(output_dir) / ('upstream-SOURCES.md' if name == source_record else name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as out:
            out.write(data)


def task_guidance(task: str) -> dict:
    """Return independent, attributed guidance for one exact DexJoCo task ID.

    This function does not claim that the local full corpus has been processed.
    Coverage and selected numeric references must be added by the corpus build.
    """
    if task not in _TASKS:
        raise ValueError(f'unknown DexJoCo task: {task}')
    value = _TASKS[task]
    task_source = f'{SOURCE_BASE}/{task}/payload/TASK.md'
    sources = [dict(path=name, url=f'{SOURCE_BASE}/{task}/payload/{name}', sha256=digest)
               for name, digest in zip(('TASK.md', 'SOURCES.md'), _SOURCE_HASHES[task])]
    phases = [dict(id=key, name=name, instruction=instruction,
                   observable_checks=list(checks), source=task_source,
                   evidence_status='paper_demonstration_interpretation')
              for key, name, instruction, checks in value['phases']]
    recovery = [dict(symptom=symptom, response=response, source=task_source,
                     evidence_status='demonstration_derived_hypothesis')
                for symptom, response in value['recovery']]
    return dict(
        task=task, instruction=value['instruction'],
        action_contract=DUAL_CONTRACT if task.startswith('bimanual_') else SINGLE_CONTRACT,
        guidance=value['guidance'], phases=phases, recovery=recovery,
        provenance=dict(
            method='curated_paraphrase_of_published_task_knowledge',
            attribution='Joint Research Team; Has GPT-6-Astra Achieved Embodied Intelligence?; SenseNova RoboRSI Packages',
            paper='https://arxiv.org/html/2610.22471v2',
            package_repository=PACKAGE_REPOSITORY, package_revision=PAPER_PACKAGE_REVISION,
            package_version='dexjoco/v1.0.0', source_files=sources,
            dataset='DexJoCo/DexJoCo-Datasets-Raw', dataset_revision=DATASET_REVISION,
            regime='rand_full', dataset_prefix='dexjoco_raw_datasets_rand_full',
            semantic_phase_alignment='not_assigned_to_new_corpus_frames',
            recovery_validation='not_tested_in_project_simulator_or_agent_rollouts',
            new_model_synthesis=False,
            limitations=[
                'Source descriptions are training-derived guidance; they are not current observations or native success labels.',
                'No source reference IDs are assumed to equal local sorted corpus IDs.',
                'Kinematic change windows do not establish semantic phase boundaries or contact states.',
                'Published author helpers are not executed or reimplemented by this catalogue.',
            ],
            license_notice=(
                'Component-specific upstream terms apply. Retain the original package LICENSE, '
                'LICENSE-MIT, THIRD_PARTY_NOTICES.md, third_party notices and per-task source records; '
                'do not relicense copied materials as project-generated data.'),
        ),
    )
