package main

import (
	"os"
	"path/filepath"
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
