// lilith.go — Lilith data infrastructure entry point
//
// Lilith is the data infrastructure layer for Molequla.
// She breaks the closed loop: organisms no longer grow only on
// seed corpus + DNA exchange. Lilith feeds them from the outside world.
//
// Architecture:
//   lilith.aml (AML brain, runs in ariannamethod.c runtime)
//     → named pipes (FIFO)
//       → INDEX1..4 (this Go code)
//         → Reddit / RSS / Wikipedia / Git repos
//           → shared SQLite (sqlite-vec + FTS5)
//             → Mycelium → organisms
//
// The human is present but not necessary.
//
// "Та, которая была до Евы. Аминь."

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	nodeID := flag.Int("id", 1, "INDEX node ID (1-4)")
	createPipes := flag.Bool("create-pipes", false, "create FIFOs before listening")
	flag.Parse()

	if *nodeID < 1 || *nodeID > 16 {
		fmt.Fprintf(os.Stderr, "invalid node ID: %d (must be 1-16)\n", *nodeID)
		os.Exit(1)
	}

	node := NewIndexNode(*nodeID)

	if *createPipes {
		if err := node.CreatePipes(); err != nil {
			log.Fatalf("failed to create pipes: %v", err)
		}
		log.Printf("[INDEX %d] pipes created: %s, %s", *nodeID, node.CmdPipe, node.RspPipe)
	}

	// Handle graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Printf("[INDEX %d] signal received, shutting down", *nodeID)
		node.Close()
		os.Exit(0)
	}()

	log.Printf("[INDEX %d] opening pipes...", *nodeID)
	if err := node.Open(); err != nil {
		log.Fatalf("failed to open pipes: %v", err)
	}
	defer node.Close()

	log.Printf("[INDEX %d] listening for commands from Lilith", *nodeID)
	node.Listen()
	log.Printf("[INDEX %d] shutdown complete", *nodeID)
}
