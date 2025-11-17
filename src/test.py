from . import game
import torch
import sys
import numpy as np

def print_board(player_bits, opponent_bits):
    print_str = "\t0 1 2 3 4 5 6 7\n"
    for i in range(8):
        print_str += f"{i}\t"
        for j in range(8):
            pos = np.uint64(1) << np.uint64(i * 8 + j)
            if player_bits & pos:
                print_str += "X "
            elif opponent_bits & pos:
                print_str += "O "
            else:
                print_str += ". "
        print_str += "\n"
    print(print_str, end="")
        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = sys.argv[1].strip() if len(sys.argv) > 1 else "best_model.pth"

    with torch.inference_mode():
        model = game.OthelloNet()
        model.eval()
        model.to(device)
        model.compile(mode="reduce-overhead")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        root, hashtable = game.starting_node(model)

        while True:
            if len(root.children) == 0:
                print("No valid moves available. Game over.")
                player_end, ai_end = root.player, root.opponent
                break
            print("Current board:")
            print_board(root.player, root.opponent)
            if len(root.children) == 1:
                print("Only one valid move available. Auto-selecting it.")
                child_player, child_opponenet, _ = root.children[0]
            else:
                while True:
                    move_index = input("Enter your move:")
                    try:
                        [row,col] = list(map(int, move_index.split(',')))
                        move_index = row * 8 + col
                        possible_moves = game.candidates(root.player, root.opponent)
                        if possible_moves & (np.uint64(1) << np.uint64(move_index)):
                            break
                        else:
                            print("Invalid move. Try again.")
                    except:
                        print("Invalid input. Try again.")
                child_opponenet, child_player = game.move(root.player, root.opponent, np.uint64(1) << np.uint64(move_index))

            print("Board after your move:")
            print_board(child_opponenet, child_player)

            if (child_player, child_opponenet) in hashtable:
                root = hashtable[(child_player, child_opponenet)]
            else:
                root = hashtable[(child_player, child_opponenet)] = game.Node(child_player, child_opponenet, model)
            
            if len(root.children) == 0:
                print("No valid moves available. Game over.")
                ai_end, player_end = root.player, root.opponent
                break

            hashtable = game.reconstruct_hashtable(root, hashtable)
            for _ in range(256):
                root.select_batch(model, 32, hashtable)
            root = root.select_optimal(model, hashtable, temperature=0.0)
    
    ai_count = game.popcount(ai_end)
    player_count = game.popcount(player_end)
    print("Final board:")
    print_board(player_end, ai_end)
    print(f"Game over. You: {player_count}, AI: {ai_count}")