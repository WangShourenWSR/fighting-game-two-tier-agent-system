# Example file showing a basic pygame "game loop"
import pygame
import retro

import numpy as np

def renderNumpyBuffer(buffer):
    surf = pygame.surfarray.make_surface(np.rot90(np.flip(buffer, 0), k=3))
    return surf

def main():
    env = retro.make(
                        game='StreetFighterIISpecialChampionEdition-Genesis', 
                        scenario='multiplayer', 
                        players=2, 
                        use_restricted_actions=retro.Actions.ALL
                    )
    obs = env.reset()
    pygame.init()
    original_size = (320, 224)
    screen = pygame.display.set_mode(original_size, pygame.RESIZABLE)
    clock = pygame.time.Clock()
    running = True
    while running:
        events = pygame.event.get()
        keys = pygame.key.get_pressed()
        
        actions = env.action_space.sample()
        #actions = np.zeros(24)
        actions[0:12] = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        actions[0]  = keys[pygame.K_x]     # B
        actions[1]  = keys[pygame.K_z]     # A
        actions[2]  = keys[pygame.K_F2]    # MODE
        actions[3]  = keys[pygame.K_F1]    # START
        actions[4]  = keys[pygame.K_UP]    # UP
        actions[5]  = keys[pygame.K_DOWN]  # DOWN
        actions[6]  = keys[pygame.K_LEFT]  # LEFT
        actions[7]  = keys[pygame.K_RIGHT] # RIGHT
        actions[8]  = keys[pygame.K_c]     # C
        actions[9]  = keys[pygame.K_s]     # Y
        actions[10] = keys[pygame.K_a]     # X
        actions[11] = keys[pygame.K_d]     # Z

        actions[14] = 0 # P2 MODE
        actions[15] = 0 # P2 START

        # action_space will by MultiBinary(16) now instead of MultiBinary(8)
        # the bottom half of the actions will be for player 1 and the top half for player 2
        obs, rew, done, info = env.step(actions)
        # rew will be a list of [player_1_rew, player_2_rew]
        # done and info will remain the same
        
        screen.fill((0, 0, 0)) # Fill with all black pixels
        image_buffer = renderNumpyBuffer(env.get_screen())
        
        image_buffer = pygame.transform.scale(image_buffer, (screen.get_width(), screen.get_height()))
        screen.blit(image_buffer, (0,0))
        
        #surface = renderNumpyBuffer(env.get_screen())
        #pygame.draw.rect(surface, 
        #                    (0,0,0), 
        #                    (surface.get_width()/3, surface.get_height()/3, surface.get_width()/3, surface.get_height()/3)
        #                )
        #pygame.display.flip()
        pygame.display.update()
        
        clock.tick(60)
        
        if done:
            obs = env.reset()
            
    env.close()
    
    pygame.quit()

if __name__ == "__main__":
    main()