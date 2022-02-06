import pygame
import os.path as path


win = pygame.display.set_mode((800, 600))
grid_file_path = 'grid.png'
grid_background = pygame.image.load(grid_file_path)
win.fill((64, 224, 208))
win.blit(grid_background, (135, 35))

while True:
    pygame.display.set_caption("Simple Maze Environment")
    pygame.display.flip()



