package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The senses (phone1/senses.sh) write into ../dna/output/{world,sound,place}
// on their own schedule, hours before an organism wakes. Those directories are
// only food if the organism was told about them: CFG.DNAExtraSources is nil by
// default, so without the flag every fragment the camera, the microphone and
// the place left behind is invisible. These cases drive the real parseCLIArgs,
// not a copy of it.

// withArgs runs f with os.Args replaced and CFG restored afterwards.
func withArgs(t *testing.T, args []string, f func()) {
	t.Helper()
	savedArgs := os.Args
	savedCFG := CFG
	defer func() { os.Args = savedArgs; CFG = savedCFG }()
	os.Args = append([]string{"molequla_cgo"}, args...)
	f()
}

func TestDNAExtraSourcesFlagReachesTheField(t *testing.T) {
	withArgs(t, []string{"--element", "earth", "--dna-extra-sources", "world,sound,place"}, func() {
		CFG.DNAExtraSources = nil
		if _, _, element, _ := parseCLIArgs(); element != "earth" {
			t.Fatalf("element %q, want earth", element)
		}
		if len(CFG.DNAExtraSources) != 3 {
			t.Fatalf("CFG.DNAExtraSources = %v, want three entries", CFG.DNAExtraSources)
		}
		srcs := dnaSources("earth")
		in := func(want string) bool {
			for _, s := range srcs {
				if s == want {
					return true
				}
			}
			return false
		}
		for _, want := range []string{"world", "sound", "place"} {
			if !in(want) {
				t.Fatalf("dnaSources(earth) = %v, missing %q", srcs, want)
			}
		}
		// The elements are still there, and earth still does not eat itself.
		for _, want := range []string{"air", "water", "fire"} {
			if !in(want) {
				t.Fatalf("dnaSources(earth) = %v, missing the element %q", srcs, want)
			}
		}
		if in("earth") {
			t.Fatalf("dnaSources(earth) = %v, an organism must not eat its own emissions", srcs)
		}
	})
}

func TestDNAExtraSourcesFlagIsTolerantOfSpacingAndEmptyNames(t *testing.T) {
	withArgs(t, []string{"--dna-extra-sources", " world , ,sound "}, func() {
		CFG.DNAExtraSources = nil
		parseCLIArgs()
		if len(CFG.DNAExtraSources) != 2 ||
			CFG.DNAExtraSources[0] != "world" || CFG.DNAExtraSources[1] != "sound" {
			t.Fatalf("CFG.DNAExtraSources = %q, want [world sound]", CFG.DNAExtraSources)
		}
	})
}

func TestDNAExtraSourcesAbsentByDefault(t *testing.T) {
	withArgs(t, []string{"--element", "earth"}, func() {
		CFG.DNAExtraSources = []string{"stale"}
		parseCLIArgs()
		if len(CFG.DNAExtraSources) != 1 || CFG.DNAExtraSources[0] != "stale" {
			t.Fatalf("CFG.DNAExtraSources = %v, an absent flag must change nothing", CFG.DNAExtraSources)
		}
	})
}

// End to end over the flag: what senses.sh writes into dna/output/world is what
// dnaListNew hands the organism.
func TestDNAExtraSourcesFragmentIsListedForTheOrganism(t *testing.T) {
	root, restore := dnaTestTree(t, "world", "sound", "place")
	defer restore()

	withArgs(t, []string{"--element", "earth", "--dna-extra-sources", "world,sound,place"}, func() {
		CFG.DNAExtraSources = nil
		parseCLIArgs()

		frag := "[eye cam0 2026-09-13T22:12:01Z] A blurry kitchen table shows a green bowl, a spoon, and a plate."
		dnaWriteFragment(t, root, "world", "gen_1789337521_2.txt", frag)
		dnaWriteFragment(t, root, "place", "gen_1789337452_1.txt", "[place 2026-09-13T22:10:52Z] The phone is at Neve Menachem.")

		if err := os.Chdir(filepath.Join(root, "earth")); err != nil {
			t.Fatal(err)
		}
		seen := map[string]int{}
		for _, src := range dnaSources("earth") {
			seen[src] = len(dnaListNew(filepath.Join("../dna/output", src), ""))
		}
		if seen["world"] != 1 {
			t.Fatalf("dnaListNew saw %d fragments in world, want 1 (sources: %v)", seen["world"], dnaSources("earth"))
		}
		if seen["place"] != 1 {
			t.Fatalf("dnaListNew saw %d fragments in place, want 1", seen["place"])
		}
		if seen["sound"] != 0 {
			t.Fatalf("dnaListNew invented %d fragments in an empty sound", seen["sound"])
		}
	})
}

// Routing repair 2. The siblings emit a fragment per tick each and the senses
// write a few an hour, so a backlog of sibling chatter is the normal state of
// the field. Under one shared budget the extra sources, which dnaSources lists
// last, were read only when the siblings happened to leave room; this case is
// the shape of that starvation — 64 sibling fragments standing in front of one
// world fragment — and the world fragment has to be in the corpus after a
// single dnaRead. Red under one budget: the eight reads are spent inside air.
func TestDNAExtraSourcesReadUnderTheirOwnBudget(t *testing.T) {
	root, restore := dnaTestTree(t, "world", "sound", "place")
	defer restore()

	const worldText = "[eye cam0 2026-09-13T22:12:01Z] A blurry kitchen table shows a green bowl and a spoon."
	for i := 0; i < 64; i++ {
		dnaWriteFragment(t, root, "air",
			fmt.Sprintf("gen_1789337000_%d.txt", i),
			fmt.Sprintf("air fragment %d, long enough to be food for a sibling.", i))
	}
	dnaWriteFragment(t, root, "world", "gen_1789337521_2.txt", worldText)

	withArgs(t, []string{"--element", "earth", "--dna-extra-sources", "world,sound,place"}, func() {
		CFG.DNAExtraSources = nil
		parseCLIArgs()
		if err := os.Chdir(filepath.Join(root, "earth")); err != nil {
			t.Fatal(err)
		}
		corpus := filepath.Join(root, "earth", "nonames_earth.txt")
		if err := os.WriteFile(corpus, []byte("seed line.\n"), 0644); err != nil {
			t.Fatal(err)
		}
		cur := &dnaCursor{Last: map[string]string{}}
		if added := dnaRead("earth", corpus, nil, nil, cur); added == 0 {
			t.Fatal("dnaRead ate nothing at all")
		}
		body, err := os.ReadFile(corpus)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(body), worldText) {
			t.Fatalf("one dnaRead behind 64 sibling fragments left the world fragment uneaten; "+
				"corpus holds %d lines, cursor: %v",
				strings.Count(string(body), "\n"), cur.Last)
		}
		// The siblings keep their own budget: the world fragment must not have
		// been bought with sibling reads.
		if n := strings.Count(string(body), "air fragment "); n != CFG.DNAMaxReadsPerTick {
			t.Fatalf("air was read %d times, want CFG.DNAMaxReadsPerTick = %d", n, CFG.DNAMaxReadsPerTick)
		}
	})
}

// And the mirror case: a flood of senses fragments must not eat the siblings'
// budget either. Both halves are bounded, neither is entitled to the other's.
func TestDNAExtraSourcesCannotStarveTheSiblings(t *testing.T) {
	root, restore := dnaTestTree(t, "world", "sound", "place")
	defer restore()

	for i := 0; i < 64; i++ {
		dnaWriteFragment(t, root, "world",
			fmt.Sprintf("gen_1789337000_%d.txt", i),
			fmt.Sprintf("[eye cam0] world fragment %d, a whole sentence of it.", i))
	}
	dnaWriteFragment(t, root, "air", "gen_1789337521_2.txt", "air said something worth eating.")

	withArgs(t, []string{"--element", "earth", "--dna-extra-sources", "world,sound,place"}, func() {
		CFG.DNAExtraSources = nil
		parseCLIArgs()
		if err := os.Chdir(filepath.Join(root, "earth")); err != nil {
			t.Fatal(err)
		}
		corpus := filepath.Join(root, "earth", "nonames_earth.txt")
		if err := os.WriteFile(corpus, []byte("seed line.\n"), 0644); err != nil {
			t.Fatal(err)
		}
		dnaRead("earth", corpus, nil, nil, &dnaCursor{Last: map[string]string{}})
		body, err := os.ReadFile(corpus)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(body), "air said something worth eating.") {
			t.Fatal("64 world fragments swallowed the sibling's turn")
		}
		if n := strings.Count(string(body), "world fragment "); n != CFG.DNAExtraReadsPerTick {
			t.Fatalf("world was read %d times, want CFG.DNAExtraReadsPerTick = %d", n, CFG.DNAExtraReadsPerTick)
		}
	})
}
