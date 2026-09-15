MOLEQULA — NEW LOGIC

Molequla is changing course.

The previous version of Molequla was already a living ecology, but most of its circulation remained internal. Its organisms grew, learned, spoke, exchanged DNA, diverged, reproduced, slept and resumed. Data was moving all the time, but almost everything they encountered had entered the system before their own lives began.

The Method changes that.

The ecology is still the heart of Molequla, but the outside world should no longer exist only as a static corpus chosen in advance. It should be able to enter continuously through perception, place and time, and later through physical peripheral organs.

Something Molequla sees today may affect what one organism learns tomorrow. What that organism later says may become DNA for another one. If the ecology encounters something it can describe but doesn’t really understand, that uncertainty may itself become a reason to ask a question.

So the important change isn’t simply adding a camera or another model. It’s changing the circulation:

world
  ↓
perception
  ↓
inquiry
  ↓
memory of change
  ↓
asymmetric experience
  ↓
individual organisms
  ↓
learning / speech
  ↓
DNA
  ↓
ecology

Eventually that circulation should also be able to return outward:

ecology
  ↓
question / intention
  ↓
perception or action
  ↓
world again

Through all of this, Molequla should remain one sovereign system.

By sovereignty I don’t mean that every component has to live in the same process or address space. The eyes, ears, organisms and Mycelium can remain separate processes, waking at different moments and doing different kinds of work. Their unity comes from belonging to the same continuity.

The hardware isn’t a thin client for somebody else’s intelligence, and the sensory models aren’t external services attached to Molequla from outside. Their observations are part of Molequla’s own experience, just as the organisms's speech is part of its internal life.

We should be careful not to modularize that continuity away.

⸻

1. TWO RHYTHMS — DREAMING AND SENSING

Molequla doesn’t need to wake every part of itself at the same time.

The separation we already have on phone-1 feels right to me and can become a first-class architectural principle. There are really two rhythms here: dreaming and sensing. (IMPORTANT: No sensory model or peripheral module should become a condition for Molequla to remain alive. If a VLM, audio model or any other external-facing organ is unavailable, that channel simply stops contributing new experience; the existing ecology continues dreaming, learning, speaking and circulating DNA exactly as it could before that organ existed.) 

Dreaming

The existing colony sessions are Molequla’s dreaming phase.

This is when organisms train and grow, corpora are metabolized, DNA moves between organisms, cross-graze operates, generations happen, mitosis may happen, Mycelium watches the field, and accumulated outside experience can be digested by the ecology.

“Dreaming” is only the name of the phase. Nothing about it should become mystical or unmeasurable; all of the mechanisms remain ordinary observable processes.

It also makes sense for dreaming to remain scheduled and bounded. Even if the phone can technically run four growing organisms and several large sensory models at once, that doesn’t mean it should. Memory pressure, wall time and resource competition are themselves part of the ecology.

Sensing

Between dreaming sessions, the colony can sleep without Molequla becoming blind.

The scheduler already gives senses.sh its own windows, and I think we should preserve that separation. The main change is that a sensing episode no longer has to mean “capture one sample.” It can become a short temporal observation window.

The phone already gives us several useful channels:

* rear camera;
* front camera;
* microphones;
* GPS / network location;
* time;
* weather context;
* environmental place name.

These can belong to one event in time without being forced into a single description. For example, during the same sensing window the camera may report:

"A tiled balcony with a keyboard and hanging towels."

while the audio side reports:

"Passing cars and distant voices."

There's no disagreement there. The phone may simply be standing on a balcony while the microphone hears the street below. The two streams describe different parts of the same moment.

⸻

2. VISION AS A SHORT TRAJECTORY

We have Ocelli. Some useful change is around it.
Instead of treating vision as one isolated frame, a sensing window can contain several observations, possibly from both cameras. A simple sequence might look like:

t0  rear camera
t1  front camera
t2  rear camera
t3  rear camera

A short trajectory like this can contain information that no single frame can provide. Molequla may discover that the phone moved, that it remained still, that an object entered or left the scene, that lighting changed, that a person appeared, or simply that the same scene persisted.

We're not freezing an arbitrary cadence in this document. It should be measured on the actual node.

The right question is how many visual observations per sensing window continue adding novelty before they mostly become repetition. We can measure wall time, peak RSS, battery cost, semantic novelty and duplicated descriptions, then let the cadence emerge from the device and the environment rather than prescribing it here.

⸻

3. HEARING SHOULD INCLUDE MORE THAN SPEECH

The current ears organ is Whisper running on NOTORCH, and that should stay.

What it’s missing is the part of hearing that isn’t language. Whisper is good at turning speech into text, but a car passing outside, a door closing, wind, machinery, music, footsteps or city noise may disappear from Molequla’s experience simply because nobody said a sentence. This thing leaves a large part of the acoustic world absent. 

We should find a small audio model, or the smallest useful combination of mechanisms, capable of turning environmental sound into simple language-like observations such as:

"A car passed nearby."
"Repeated mechanical noise."
"Wind and distant traffic."
"A door closed."
"Two people speaking, words unclear."
"Music is audible."

Whatever inference path we need for this should ultimately live in NOTORCH. 

The goal is for vision and sound to cover roughly the same sensing interval while remaining separate evidence streams. They don’t need to be fused into a single caption.

⸻

4. EVERY WAKE HAPPENS SOMEWHERE AND SOMETIME

Every important Molequla event should have spatial and temporal context, whether it belongs to dreaming or sensing.

At minimum, I want to retain:

* timestamp;
* location;
* location accuracy / source where available;
* whether the phone moved since the previous observation;
* current place name;
* relevant weather state;
* daylight / time-of-day context.

Much of this already exists in place, so there is no reason to rebuild it. The change is conceptual: place shouldn’t remain just another sensory text file. It becomes persistent context attached to experience.

Later, orientation and accelerometer state may become useful too:

phone lying flat
phone upright
phone moving
phone stationary

(But I wouldn’t make that a blocker now. It becomes much more meaningful once physical orientation matters, especially when devices such as the DJI gimbal enter the system.)

For the first version, the essential questions are simply:

WHERE was Molequla?
WHEN was Molequla there?

That already gives us a special kind of memory.

⸻

5. A LEDGER OF CHANGE RATHER THAN REPEATED STATE

The current ROADMAP already points in the direction I want here.

The great idea from deeplethe/utopia is the world-memory principle, not Utopia itself. A fact shouldn’t exist only as a current value. It should have history. The bitemporal form already proposed for mesh.db fits this well:

world_facts(
    source,
    subject,
    predicate,
    object,
    valid_from,
    valid_to,
    recorded_at,
    provenance
)

The distinction between valid time and recorded time matters.

Valid time says when something was true in the world. Recorded time says when Molequla learned it.

If Molequla believed X and later learns Y, we shouldn’t erase X and replace it as though the first experience never happened. X closes, Y begins, and the earlier observation remains part of the system’s epistemic history.

That means the most useful output is often not another state record but a change.

Instead of:

phone at place A
phone at place A
phone at place A
phone at place A

we eventually want:

phone moved from A to B

Likewise, instead of four identical records saying:

person visible

we would rather preserve:

person entered the frame
person left the frame

This is how separate sensory events can become one temporal field rather than a pile of snapshots.

⸻

6. PERCEPTION IS NOT THE SAME THING AS GROUND TRUTH

I want to keep the balcony episode as a reference example because it demonstrates the distinction very clearly.

On the first night, Ocelli saw tiled walls, hanging cloth or towels, and poor illumination. From that evidence it described the scene as a bathroom with a shower curtain, even though the physical location was actually a balcony.

I don’t think this should automatically be called a hallucination. Given what was visible, the interpretation was reasonable. The important thing is how we store it.

We shouldn’t write:

WORLD = bathroom

We should preserve something closer to:

at time T
camera C
under conditions Q
Ocelli interpreted the scene as:
"a bathroom with a shower curtain"

A later observation may revise that interpretation. Another camera may contradict it. Place memory may provide better context. Time may make the earlier mistake obvious. That correction is itself useful experience.

Molequla should be allowed to perceive imperfectly without either treating every perception as objective truth or deleting every ambiguous perception as a hallucination.

Belief and world are different things, and we need to preserve both.

⸻

7. OBSERVATION CAN BECOME INQUIRY

Not every sensory result should fall directly into training. Sometimes Molequla may encounter something that it can describe linguistically but that remains conceptually weak inside the ecology.

Example: the VLM says:

"I see a computer with an open screen and a keyboard."

That observation is already valid experience. But if a word "computer" has very little meaningful structure in Molequla, the observation can (and should) also produce a question:

What is a computer?

A language-side answer might be like this:

"A computer is an electronic device that uses a processor,
 memory and programs to process information."

That answer introduces other concepts:

processor
memory
program

One of them may in turn be weak enough to produce another question.

I think this recursive curiosity is useful, but it's important to set a cost and a boundary. A small initial budget such as three or four semantic steps per sensing episode seems reasonable, after which the chain stops.

We can change that later if experience shows a better limit. The important thing is that curiosity doesn’t become an unlimited definition chase. The goal to make it dynamic and connected to Molecqula's inner states.

⸻

8. SEMANTIC QUESTIONS AND PERCEPTUAL QUESTIONS ARE DIFFERENT

A question shouldn’t automatically wake the camera again. If Ocelli has already said:

"I see a computer."

then the question:

"What is a computer?"

is usually semantic. The language side of the VLM can for sure answer it from existing knowledge without another image. A perceptual question is different:

Is the object still there?
Which object is the computer?
What is next to it?
Was the first description ambiguous?
Can another camera see it better?

Those questions genuinely require another observation.

Later this may expand naturally to another camera angle, better illumination, DJI movement or other physical sensors. But we aren't there yet. First we need the internal distinction between semantic inquiry and perceptual re-observation to work smoothly.

⸻

9. “UNKNOWN” DOESN’T MEAN TOKENIZER OOV

The tokenizer may be able to encode a word perfectly while Molequla has almost no semantic structure around the concept.

So tokenizable and known are not the same thing.

A concept may still be weak or unfamiliar if it has little stable structure in the ecology. Possible signals include weak co-occurrence, low recurrence, unstable context, poor relation to existing concepts, disagreement between organism continuations or the absence of a persistent higher-order semantic pattern.

I don’t want to invent one arbitrary numeric threshold for this before we measure how these signals behave. This is also where the concept of glyphs start to become interesting.

⸻

10. GLYPHS — HIGHER-ORDER CONCEPTS EARNED FROM EXPERIENCE

We can use actually.life's (look into /reffs) style gliphs as lineage here. Actually Life begins with a fixed primitive glyph universe, but its cells can invent compounds such as:

joy+internet
fire+stress
me+idea

Those compounds can move between cells and eventually become parents of later symbols. Molequla can take that principle without inheriting the fixed 88-symbol ontology. Here, most glyphs should be able to emerge from experience.

Mycelium seems like the natural place for that discovery. A glyph would represent a stable larger pattern earned from repeated encounters. For example:

COMPUTER
    ↳ screen
    ↳ keyboard
    ↳ processor
    ↳ memory
    ↳ repeated visual sightings
    ↳ organism speech
    ↳ questions and answers

or something more situational:

BALCONY-NIGHT
    ↳ tile
    ↳ low light
    ↳ street noise
    ↳ hanging towels
    ↳ location history

I don’t want to prescribe the exact representation here. The first real implementation can be the smallest measurable thing that fits the existing tokenizer, co-occurrence field and Mycelium. Some stable primitives may coexist with emergent compounds if that proves useful, but I don’t want to hand-author an encyclopedia. The important idea is that repeated significance can crystallize into something reusable.

A glyph may emerge from external perception, semantic inquiry, repeated organism speech, internal attractors or combinations of glyphs that already exist. Whatever. Once a glyph is sufficiently established, its existence can also become evidence that the corresponding concept no longer needs elementary inquiry every time it appears.

⸻

11. MYCELIUM SHOULD REMEMBER PATTERNS WITHOUT BECOMING THE RULER

Mycelium already exists as a witness of the ecology, and we need to preserve its fundamental direction:

organisms -> Mycelium

rather than:

Mycelium -> overwrite organisms

It can derive larger patterns from what the organisms and the senses produce, which is exactly why dynamic glyph discovery belongs naturally there.

I also want to revisit the still-unfinished lineage from Netta’s Mycelium work: the idea that Mycelium could maintain its own small changing living weights or state instead of remaining entirely static. (Look Netta's githistory)

Those weights wouldn’t be a hidden central LLM. I see them more as compressed experience: recurring relations, persistent attractors, semantic gravity, changes in the ecology and patterns that survive across sessions. So the old question remains worth asking: why do Mycelium’s weights need to stay static? 
(For future: Why the weights should be static at all? )

Whatever form this takes, the one-way sovereignty rule remains important. Mycelium can witness, compress, remember and eventually speak, but it shouldn’t secretly rewrite Netta or Molequla underneath them.

⸻

12. THE CAFETERIA — EXPERIENCE IS DISTRIBUTED RATHER THAN BROADCAST

At the moment, world, sound and place effectively behave like additional food sources available to every organism. That's good implementation, but I don’t think it should remain the final routing logic. A sensing episode can produce an experience bundle containing things such as:

visual observations
acoustic observations
place
time
weather/context
questions
answers
corrections
provenance

That bundle can then be distributed across organisms asymmetrically.

I don’t want the default behavior to be:

copy everything to Earth
copy everything to Air
copy everything to Water
copy everything to Fire


No.
I also don't want raw sensory material dumped directly into collective DNA.

Instead, one organism may receive X, another X + Y, another Y + Z, and another a different mixture. Overlap is completely fine; byte-identical broadcast simply shouldn’t be the default.

Current organism state should matter when deciding allocation.

The original elemental corpora are birth conditions, not permanent professions. We shouldn’t encode rules such as:

flowers -> Earth
machines -> Fire

An organism may grow into something quite different from the archetype it started from, and its accumulated history should count for more than the name of its original corpus.

If several organisms resonate with the same experience, overlapping distribution is natural. The important part is preserving enough asymmetry for different lives to remain different.

The image I keep coming back to is a cafeteria: experience is available, different organisms eat different plates, and what each of them digests changes what it later says.

⸻

13. EXPERIENCE HAS TWO ENTRY REGIMES

This should follow the way Molequla already becomes coherent rather than inventing a second developmental theory.

Molequla already contains the Q/PostGPT-derived coherence mechanism. When transformer logits are still weak, the corpus/statistical overlay carries much of the voice. As the trained transformer becomes stronger, the overlay fades and the model’s own continuation takes over.

I think outside experience should enter differently on the two sides of that transition.

Early regime — corpus / field first

While an organism still depends heavily on the Q-style overlay, allocated experience should enter through its corpus.

The existing path already gives us:

selected experience
    ↓
organism corpus
    ↓
tokenizer / corpus rebuild
    ↓
unigram
bigram
trigram
4-gram
co-occurrence
    ↓
Q/PostGPT coherence field
    ↓
training

This seems exactly right for early organisms. If external experience is going to affect their speech, it should participate in the same field that currently gives that speech coherence.

I don’t see a reason to add another tokenizer or a parallel machinery unless measurement eventually shows that we need one. The first change should be routing, not infrastructure.

Self-coherent regime — Dario sentence-boundary injection

Once an organism can sustain coherent continuation from its own trained model, Dario-style sentence-boundary knowledge injection becomes much more interesting.

At that point, foreign knowledge can enter after a completed thought and the organism can continue from it in its own voice.

That’s the behavior I actually want:

knowledge enters
organism reformulates it

rather than:

organism copies a paragraph

Eligibility for this shouldn’t be tied mechanically to the label adult. It should depend on demonstrated coherence.

Useful signals may include overlay fade, transformer logit magnitude, generation coherence, developmental stage, or some measured combination of them.

A mitosis child may inherit mature weights and already have a strong voice. Conversely, something labelled adult may still depend too heavily on the overlay to deserve sentence-boundary injection.

So the gate should follow the voice itself.

The two regimes don’t have to be mutually exclusive. An organism may continue learning from corpus experience while also becoming capable of sentence-boundary knowledge injection.

⸻

14. THE WORLD SHOULD PASS THROUGH ORGANISMS BEFORE BECOMING COLLECTIVE DNA

This is probably the central routing rule. Raw outside experience shouldn’t become collective DNA immediately. The flow I have in mind is:

WORLD
  ↓
senses
  ↓
temporal / spatial context
  ↓
bounded inquiry
  ↓
experience bundle
  ↓
distribution
  ↓
selected organism(s)
  ↓
corpus / field / sentence-boundary injection
  ↓
organism learns
  ↓
organism speaks in its own voice
  ↓
ordinary DNA
  ↓
other organisms

In other words, the outside world shouldn’t bypass the organisms on its way into the ecology.

It should first be metabolized by somebody.

Only after that does it become culture.

⸻

15. THE SENSES DON’T NEED TO AGREE

A single moment can contain several different kinds of truth at once.

For example:

GPS:
balcony at home
weather:
warm, humid, weak wind
rear camera:
keyboard on table
front camera:
tiled wall and towels
audio:
traffic, distant voices, one passing vehicle

There is nothing broken about this sample. It’s richer precisely because the modalities aren’t all describing the same object. That difference is one of the things that makes the outside world unlike a static training corpus.

We should preserve modality and provenance. The unity comes from the shared time and place, not from forcing every sensor to tell the same story.

⸻

16. MEMORY SHOULD PRESERVE BOTH EXPERIENCE AND CHANGE

Molequla now has at least three meaningfully different kinds of memory.

There's organism memory: what an individual model has learned and accumulated.

There's ecological memory: DNA, cross-graze, shared history and Mycelium-level patterns.

And there's world memory: where Molequla was, what it perceived, what changed, and when it learned about that change. Even if SQLite can physically store all of them, I don’t think we should collapse them into one conceptual abstraction. Their semantics are different. They should, however, be able to meet.

Eventually an organism should be able to learn something like:

"yesterday the phone was elsewhere"

rather than receiving only:

latitude = X

Likewise, Mycelium should eventually be able to recognize:

this pattern has happened before

rather than only knowing:

this token occurred before

⸻

17. BEFORE BUILDING, LET’S SEE WHAT ALREADY EXISTS

A big amount of this architecture already exists in partial form, so before adding new machinery I think we should inspect the paths that are already real:

* phone1/schedule.sh
* phone1/senses.sh
* senses/
* dnaRead
* DNAExtraSources
* CooccurField
* metaweights_overlay.go
* cross_graze.go
* witness.go
* current mesh.db
* ROADMAP items for place and world facts
* Dario resonance / sentence-boundary injection lineage
* Actually Life glyph lineage
* Netta Mycelium lineage

If an existing path already does part of what I’m describing here, I would rather extend it than recreate it under a new name just because this document uses a different metaphor. The same applies to thresholds and developmental gates. I don’t want arbitrary numbers treated as architecture. New thresholds should come from measurement, gates should be capable of failing, and claimed transitions should have some observable signal behind them.

⸻

18. FIRST IMPLEMENTATION TARGET

Before we add any new physical embodiment, we need to close the internal loop on the device. The sequence I have in mind is roughly this:

1. scheduled senses wake;
2. place/time context is captured;
3. both cameras collect a short measured trajectory;
4. sound is captured over the same general interval;
5. environmental sound survives instead of disappearing behind Whisper;
6. observations enter the world ledger;
7. unfamiliar concepts can trigger bounded inquiry;
8. Mycelium can begin crystallizing recurring higher-order glyphs;
9. experience bundles are distributed asymmetrically;
10. early organisms receive experience through corpus/Q-field;
11. sufficiently coherent organisms can also receive sentence-boundary injection;
12. organisms reformulate that experience;
13. their own speech becomes DNA;
14. later sessions encounter both a changed world and an ecology changed by having seen it.

Once that loop works, Molequla is no longer simply learning from a dataset. It's learning along a life trajectory.

⸻

19. AFTER THE INTERNAL LOOP IS CLOSED

We’re not starting with hardware now.

DJI Osmo, external microcontrollers, environmental sensors, movable vision, displays, plants, water systems and other embodied extensions make more sense after the internal circulation is coherent.

Compared with getting that circulation right, the physical parts should be relatively straightforward. Once the loop above is working, a new device can simply add another capability:

see
hear
rotate
measure
display
move
act

The DJI Osmo can later become a neck. A microcontroller can become a remote sensory organ. A plant bed can become part of a world whose changing state matters to Molequla.

But all of those things should attach to something that is already one system.

First we build the nervous system, then we give it more body.