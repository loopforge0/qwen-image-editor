"""Loop mode: generate, review against a checklist, fix, until the goal is met or the rounds run out.

Built from what works elsewhere:
- a goal split into yes/no checks before anything is made, each judged against the picture (TIFA, Davidsonian Scene
  Graph: question-based checks are what make a vision critic specific);
- a reviewer that assumes the image fails and names the most visible defect, judging the pixels and not the prompt
  that made them, and that has to look at full size before passing anything (ALIGN's adversarial visual review);
- keeping the best image so far and reverting when a round makes it worse (ALIGN); one image per round, reviewed
  before anything else is made (several candidates up front can be turned back on with new_loop's candidates);
- local fixes as edits, structural failures as a new generation from a rewritten prompt (T2I-Copilot, GenArtist).

Chat apps differ in whether they show the model a server's instructions, so the protocol travels in start_loop's
result and every review's answer, not only in the server instructions. The state lives in this process: each chat
app runs its own MCP server.
"""

import os
import time

LOOPS: dict[str, dict] = {}
MAX_LOOPS = 20

GUIDE = """Loop mode: you are now both the maker and a strict reviewer. Work in rounds until every check passes or the
rounds run out. Pass loop="{id}" to generate_image, edit_image, mark_image and inspect_image.

What Qwen-Image-2.1 can do here (use whichever fits the fix):
- New image from text: generate_image. Text in the picture renders well when quoted exactly in the prompt.
- Edit one image: edit_image with images ["<id>"]: change, add, remove, recolour, restyle, change pose or expression,
  change the background, change the text on a sign.
- Point at things: mark_image draws coloured circles on an image (up to 3) and returns a marked copy; edit
  that copy and name each mark by colour: "Replace the lamp in the red circle with a potted plant; make the mug in
  the green circle blue." The circles are removed from the result. Use it whenever a fix is about one specific
  thing and words alone could point at the wrong one (one of several people, a small object, a region of a sign).
- Several references (up to 10): edit_image with images [a, b, ...], called <image1>, <image2>, ... in that order,
  each with its role: "Put the cat from <image2> on the sofa in <image1>"; "Dress the person in <image1> in the
  jacket from <image2>"; "Repaint <image1> in the style of <image2>". To keep a likeness or a product exactly,
  pass its image rather than describing it.
- Transparent background: begin the prompt with "This is an RGBA image with transparency." and end it with "The
  image has alpha channel and the background is transparent." word for word. Works for new images (stickers, icons,
  cut-outs) and for extracting something from an image: edit_image (mark it first if there are several) with
  "Extract the cat in the red circle onto a transparent background." inside those two sentences.
- Size and shape: 16:9 (1344x768) for landscape scenes, 9:16 (768x1344) for portraits and tall subjects, square
  or another ratio only when it suits the subject or is asked for (see Size in the prompt rules).
- The user can mark images in the editor too. If a tool tells you they did, look (view_draft) and include it.

How much to ask for at once (it matters more than wording):
- A fix to an existing image does 1-3 changes well. More than that and some are ignored or half done. (Composing
  a new image from several references is different: that's one request however many references, see Big jobs.)
- Never mix a whole-image change (pose, camera angle, composition, style, lighting, time of day, background) with
  local fixes in the same edit. Do the whole-image change alone, review, then the local fixes.
- Marked fixes: at most 3 marks per edit, each a different colour and one change each. More marked items in one
  edit get mixed up; split them over rounds.
- Fixes that touch the same thing (a sign's text and its colour) go together; unrelated ones are better apart.
- If an edit ignored a change, ask for it alone next time, stronger and more specific, or mark it.
- Edits drift a little each time. After three edits in a row, or when the layout is wrong, generate again from a
  rewritten prompt instead (put the failing part early and explicitly: position, size, count, exact text).

Big jobs (several inputs, several things to change, extract and combine): plan the stages before making anything and
tell the user the plan in a few lines.
- Compose in one edit_image with loop=: the scene first (<image1>), then every reference it needs (up to 10 in all),
  each named by its tag with what to take from it and where it goes: "Put the sofa from <image2> against the back
  wall right of the fireplace, the armchair from <image3> left of the fireplace, and the plant and the lamp from
  <image4> in the left corner and at the sofa's right end." The model reads all the references together, so don't
  split a composition into one edit per object.
- A reference can be a busy photo: name the thing to take ("the green sofa from <image2>"), no need to extract it
  first. Mark it with mark_image only if the photo holds several similar things.
- Prepare a part on its own first only when it has to change before it goes in (a character's style, someone's
  clothes), or when the user wants the cut-out itself. Make these without loop= (they aren't rounds), but look at
  each one (inspect_image) and redo it if it's wrong: a bad part spoils everything after it.
- Then review the composition and fix only what failed: re-add a missing or wrong thing on its own (pass its
  reference again), or a blend pass on what looks pasted in.
- To put something in an exact spot, circle the empty spot on the scene with mark_image and say "Put the bicycle
  from <image2> inside the red circle".
- Keep each part's look by passing its image every time it's added, never by describing it.
- Added things easily look pasted in: lit for another place, too bright or saturated, crisper than the photo, no
  shadow, wrong scale. After adding something, mark it and blend it in, alone in that edit: "Make the bicycle in
  the red circle look photographed in this scene: relight it with the same warm low evening sun from the right as
  the people, slightly darker and less saturated, matching the photo's softness and grain, and add its soft shadow
  on the grass; keep everything else unchanged."
- Put a large object (a bicycle, a chair) beside a person rather than in their hands: a person holding or riding
  it tends to come out oversized and in front of them.

Each round:
1. Make. Round 1: {round1}. Later rounds: the fix review_image proposes.
2. Review, as an adversary. Assume the image fails and look for why; a picky client is about to reject it.
   - Judge only what is in the pixels. The prompt says what you asked for, not what you got; don't let it fill gaps.
   - Look at full size: inspect_image on the whole image, then zoom (area=) into anything small: text, hands, faces,
     counted objects, edges. Passing a check you only saw in the small preview is not allowed.
   - Anything added to a scene: compare it with what was already there. Brighter or more saturated, lit from another
     side, sharper edges, no shadow where it touches the ground, wrong size for the people next to it: it looks
     pasted, and that is a failure (major if it catches the eye), even when it's in the right place.
   - Text: read it letter by letter, as written. Counts: count one by one and say where each one is. Positions and
     colours: say what you actually see there.
   - Then review_image with every check: passed, what you saw, and for a failing check whether the fix is "local"
     (one thing or area: edit it, marked if needed) or "global" (layout, count, pose, style, the whole scene:
     regenerate or a whole-image edit on its own). Name the single most visible defect as worst, on the checklist
     or not, and set major when a viewer would notice it at a glance. A minor nit doesn't hold the loop up.
3. Fix, as review_image proposes, within the limits above. Work from the best image so far.
4. Stop when review_image says so. Show the user the final image id, which checks pass and, honestly, anything
   that still doesn't. Don't call a near miss a pass.

Between rounds, tell the user in one short line what failed and what you're changing.

The checks for this loop ({n}):
{checks}

{prompt_rules}"""


def new_loop(goal: str, checks: list[str], max_rounds: int, candidates: int, start: list[str] | None = None) -> dict:
    lid = os.urandom(3).hex()
    loop = {"id": lid, "goal": goal, "checks": checks, "max_rounds": max_rounds, "candidates": candidates,
            "start": list(start or []),
            "made": {},  # image id -> {"kind": "generate" | "edit", "from": [ids], "round": n}
            "inspected": set(), "reviews": [], "best": None, "done": False, "started": time.time()}
    LOOPS[lid] = loop
    while len(LOOPS) > MAX_LOOPS:
        LOOPS.pop(next(iter(LOOPS)))
    return loop


def guide(loop: dict, prompt_rules: str) -> str:
    n, start = loop["candidates"], loop["start"]
    if start:
        refs = ", ".join(f"<image{i + 1}> = {r}" for i, r in enumerate(start))
        round1 = (f"start from the user's image{'s' if len(start) > 1 else ''} ({refs}): edit_image with "
                  f"images {start} and an instruction that says what to make of {'them' if len(start) > 1 else 'it'} "
                  "(\"Turn the sketch in <image1> into a photorealistic matte painting with the same composition\")")
    else:
        round1 = "generate_image from a full prompt written by the prompt rules below"
    if n > 1:
        round1 += f"; make {n} candidates with different seeds, review each, and continue from the best"
    checks = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(loop["checks"]))
    return GUIDE.format(id=loop["id"], round1=round1, n=len(loop["checks"]), checks=checks, prompt_rules=prompt_rules)


def get(lid: str) -> dict | None:
    return LOOPS.get((lid or "").strip())


def rounds_used(loop: dict) -> int:
    return len({m["round"] for m in loop["made"].values()})


def record_made(loop: dict, image_id: str, kind: str, sources: list[str]):
    # Everything made before the first review is round 1 (the candidates); every later image is a round of its own.
    rnd = 1 if not loop["reviews"] else rounds_used(loop) + 1
    loop["made"][image_id] = {"kind": kind, "from": sources, "round": rnd}


def made_note(loop: dict, image_id: str) -> str:
    left = loop["max_rounds"] - rounds_used(loop)
    return (f" Loop {loop['id']}, round {loop['made'][image_id]['round']} of {loop['max_rounds']}"
            f" ({max(left, 0)} left). Now review it as an adversary: inspect_image on {image_id} at full size (zoom into"
            f" anything small), then review_image with every check.")


def review(loop: dict, image_id: str, verdicts: list[dict], worst: str, major: bool) -> str:
    """Record a review and say what to do next."""
    checks = loop["checks"]
    if image_id not in loop["made"]:
        return (f"{image_id} wasn't made in loop {loop['id']}. Review the images you made with loop=\"{loop['id']}\": "
                f"{', '.join(loop['made']) or 'none yet'}.")
    if len(verdicts) != len(checks):
        return (f"Give a verdict for each of the {len(checks)} checks, in order (you gave {len(verdicts)}):\n"
                + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(checks)))
    passed = [bool(v.get("passed")) for v in verdicts]
    if all(passed) and image_id not in loop["inspected"]:
        return (f"Not yet: you haven't looked at {image_id} at full size. Call inspect_image on it (and zoom into the "
                "parts the checks are about), then review again. Small previews hide garbled text, extra fingers and "
                "miscounts.")
    lazy = [i + 1 for i, v in enumerate(verdicts) if v.get("passed") and len((v.get("seen") or "").strip()) < 8]
    if lazy:
        return (f"Say what you actually saw for check{'s' if len(lazy) > 1 else ''} {', '.join(map(str, lazy))} "
                "(the words as written, each counted object and where it is, the colour and place), then review again.")

    made = loop["made"][image_id]
    entry = {"image": image_id, "round": made["round"], "kind": made["kind"], "passed": passed,
             "score": sum(passed) - (0.5 if major else 0), "worst": worst.strip(), "seen": [v.get("seen", "") for v in verdicts],
             "targeted": list(loop.get("target") or []),  # what the advice that led to this image aimed at
             "fix": [("global" if str(v.get("fix", "")).lower().startswith("g") else "local") for v in verdicts]}
    loop["reviews"].append(entry)
    best = loop["best"]
    prev_best = next((r for r in loop["reviews"] if r["image"] == best), None) if best else None
    if prev_best is None or entry["score"] >= prev_best["score"]:
        loop["best"] = image_id
    best_r = next(r for r in loop["reviews"] if r["image"] == loop["best"])

    fails = [i for i, p in enumerate(passed) if not p]
    lines = [f"Review of {image_id} (round {made['round']}): {sum(passed)}/{len(checks)} checks pass."]
    if fails:
        lines.append("Failing: " + "; ".join(f"{i + 1}. {checks[i]} (saw: {entry['seen'][i] or '-'})" for i in fails))
    if worst.strip():
        lines.append(f"Most visible defect{' (major)' if major else ''}: {worst.strip()}")

    if not fails and not major:
        loop["done"] = True
        lines.append(f"Goal met. Stop here and show the user {image_id}: every check passes. Say in a line or two how "
                     "many rounds it took and what changed along the way.")
        return "\n".join(lines)

    # Candidates still waiting for review in round 1
    pending = [i for i, m in loop["made"].items() if m["round"] == made["round"]
               and i not in {r["image"] for r in loop["reviews"]}]
    if pending:
        lines.append(f"Review the other candidate{'s' if len(pending) > 1 else ''} too ({', '.join(pending)}) before "
                     "fixing anything; then continue from the best.")
        return "\n".join(lines)

    if rounds_used(loop) >= loop["max_rounds"]:
        loop["done"] = True
        still = [checks[i] for i, p in enumerate(best_r["passed"]) if not p]
        lines.append(f"That was the last round. Stop and show the user the best image, {loop['best']} "
                     f"({sum(best_r['passed'])}/{len(checks)}), and say plainly what still fails: "
                     + ("; ".join(still) if still else best_r["worst"] or "nothing on the checklist")
                     + ". Offer to keep going or to fix it by hand in the editor (Edit this, then circle the area).")
        return "\n".join(lines)

    first = made["round"] == 1 and sum(m["round"] == 1 for m in loop["made"].values()) > 1
    if first:
        lines.append(f"Best candidate: {loop['best']} ({sum(best_r['passed'])}/{len(checks)}). Continue from it.")
    elif loop["best"] != image_id:
        lost = [checks[i] for i, (a, b) in enumerate(zip(best_r["passed"], passed)) if a and not b]
        lines.append(f"This round made it worse{': it lost ' + '; '.join(lost) if lost else ''}. Go back to the best "
                     f"image, {loop['best']} ({sum(best_r['passed'])}/{len(checks)}), and try a different fix from there.")

    # Which fix. A check that has failed after two edits in a row, or three edits in a row, needs a new generation.
    # Otherwise one global fix on its own, or up to three local ones in one edit (marked when they're about one thing).
    base, lid = loop["best"], loop["id"]
    best_fails = [i for i, p in enumerate(best_r["passed"]) if not p]
    glob = [i for i in best_fails if best_r["fix"][i] == "global"]
    local = [i for i in best_fails if best_r["fix"][i] == "local"]
    recent = loop["reviews"][-2:]
    stuck = [i for i in best_fails if len(recent) == 2
             and all(r["kind"] == "edit" and i in r["targeted"] and not r["passed"][i] for r in recent)]
    edits_in_a_row = 0
    for r in reversed(loop["reviews"]):
        if r["kind"] != "edit" or r["round"] == 1:  # round 1 from the user's images is the start, not drift
            break
        edits_in_a_row += 1
    name = lambda ids: "; ".join(f"{i + 1}. {checks[i]}" for i in ids)
    if stuck or edits_in_a_row >= 3:
        why = (f"check {', '.join(str(i + 1) for i in stuck)} failed after two edits" if stuck
               else "three edits in a row start to drift")
        if loop["start"]:
            lines.append(f"Next: {why}, so start again from the user's image{'s' if len(loop['start']) > 1 else ''}: edit_image "
                         f"with images {loop['start']} (loop=\"{lid}\", a new seed) and an instruction rewritten to put the "
                         "failing parts first and explicitly. Keep what already worked.")
        else:
            lines.append(f"Next: {why}, so generate_image again (loop=\"{lid}\", a new seed) from a rewritten prompt that "
                         "puts the failing parts early and explicitly (position, size, count, exact quoted text), following "
                         "the prompt rules. Keep what already worked in the prompt.")
        loop["target"] = best_fails
    elif glob:
        lines.append(f"Next: the whole-image fix on its own, nothing else in the same request: {name(glob[:1])}. Either a "
                     f"whole-image edit_image on {base}, or, if the layout itself is wrong, generate_image with a rewritten "
                     f"prompt and a new seed (loop=\"{lid}\"). Local fixes wait for the next round.")
        loop["target"] = glob[:1]
    elif local:
        pick = local[:3]
        lines.append(f"Next: fix {'these' if len(pick) > 1 else 'this'} in one edit_image on {base} (loop=\"{lid}\"): "
                     f"{name(pick)}. One change each, nothing else. If a fix is about one specific thing or area, first "
                     f"mark_image on {base} (a circle per thing, a different colour each) and edit the marked copy, naming "
                     "each mark by colour."
                     + (f" Leave {name(local[3:])} for the next round." if len(local) > 3 else ""))
        loop["target"] = pick
    elif worst.strip():
        lines.append(f"Next: fix that defect alone with edit_image on {base} (loop=\"{lid}\"), marked if it's one spot; "
                     "everything else unchanged.")
        loop["target"] = []
    lines.append(f"{loop['max_rounds'] - rounds_used(loop)} round(s) left. Tell the user in one line what you're changing.")
    return "\n".join(lines)
