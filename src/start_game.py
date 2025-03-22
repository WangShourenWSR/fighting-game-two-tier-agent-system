from game_manager.game_manager import GameManager
import warnings

if __name__ == "__main__":
    # Ignore all UserWarnings
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    game_manager = GameManager()
    game_manager.start_game(
        blind_test= True,
    ) 