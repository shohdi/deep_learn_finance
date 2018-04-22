from __future__ import division , print_function;
import collections;
import numpy as np ;
import pygame;
import random;
import os;


class MyWrappedGame(object):
    def __init__(self):
        #run pygame in headless mode
        os.environ["SDL_VIDEODRIVER"] = "dummy";

        pygame.init();

        #set constants
        self.COLOR_WHITE=(255,255,255);
        self.COLOR_BLACK=(0,0,0);

        self.GAME_WIDTH = 400;
        self.GAME_HEIGHT = 400;
        self.BALL_WIDTH = 20;
        self.BALL_HEIGHT = 20;
        self.PADDLE_WIDTH = 50;
        self.PADDLE_HEIGHT = 10;
        self.GAME_FLOOR = 350;
        self.GAME_CEILING = 10;
        self.BALL_VELOCITY = 10;

        self.PADDLE_VELOCITY = 20;
        self.FONT_SIZE = 30;
        self.MAX_TRIES_PER_GAME=1;
        self.CUSTOM_EVENT = pygame.USEREVENT +1;
        self.font = pygame.font.SysFont("Comic Sans MS",self.FONT_SIZE);



    def reset(self):
        self.frames = collections.deque(maxlen=4);
        self.game_over = False;
        #initialize positions
        self.paddle_x = self.GAME_WIDTH //2;
        self.game_score = 0;
        self.reward = 0;
        self.ball_x = random.randint(0,self.GAME_WIDTH-self.BALL_WIDTH);
        self.ball_y = self.GAME_CEILING;
        self.num_tries = 0;

        #set up display , clock ,etc
        self.screen = pygame.display.set_mode((self.GAME_WIDTH,self.GAME_HEIGHT));
        self.clock = pygame.time.Clock();



    def step(self,action):
        pygame.event.pump();

        if action == 0:# move paddle left
            self.paddle_x -= self.PADDLE_VELOCITY;
            if self.paddle_x < 0 :
                #bounce off the wall , go right
                self.paddle_x = self.PADDLE_VELOCITY;
        elif action == 2: #mofe paddle right
            self.paddle_x += self.PADDLE_VELOCITY;
            if self.paddle_x > self.GAME_WIDTH - self.PADDLE_WIDTH:
                #bounce off the wall , go left
                self.paddle_x = self.GAME_WIDTH - self.PADDLE_WIDTH - self.PADDLE_VELOCITY;
        else : #don't move paddle
            pass;
        
        self.screen.fill(self.COLOR_BLACK);
        score_text = self.font.render("Score: {:d}/{:d}, Ball: {:d}".format(self.game_score,self.MAX_TRIES_PER_GAME,self.num_tries),True,self.COLOR_WHITE);
        self.screen.blit(score_text,((self.GAME_WIDTH - score_text.get_width()),(self.GAME_FLOOR + self.FONT_SIZE//2)));
        #update ball position
        self.ball_y += self.BALL_VELOCITY;
        ball = pygame.draw.rect(self.screen,self.COLOR_WHITE,pygame.Rect(self.ball_x,self.ball_y,self.BALL_WIDTH,self.BALL_HEIGHT));
        #update paddle position
        paddle = pygame.draw.rect(self.screen,self.COLOR_WHITE,pygame.Rect(self.paddle_x,self.GAME_FLOOR,self.PADDLE_WIDTH,self.PADDLE_HEIGHT));
        #check for collision and update reward
        self.reward = 0;


        if self.ball_y >= self.GAME_FLOOR - self.BALL_HEIGHT//2:
            if ball.colliderect(paddle):
                self.reward = 1
            else:
                self.reward = -1
            self.game_score += self.reward;
            self.ball_x = random.randint(0,self.GAME_WIDTH);
        
            self.ball_y = self.GAME_CEILING;
            self.num_tries +=1

        pygame.display.flip()

        #save last 4 frames
        self.frames.append(pygame.surfarray.array2d(self.screen));
        if self.num_tries >= self.MAX_TRIES_PER_GAME :
            self.game_over = True;
        self.clock.tick(30);
        return np.array(list(self.frames)),self.reward,self.game_over
    



