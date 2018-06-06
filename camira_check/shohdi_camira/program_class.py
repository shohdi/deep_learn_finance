
import pygame;
import pygame.mixer;
import pygame.camera;
import os;
import collections;
import numpy as np ;
from sklearn.metrics import mean_squared_error
from gtts import gTTS







#1/1 - 1/50 - 1/100 - 1/500 - 1/1000

class ProgramClass:
    def __init__(self):
        #os.environ["SDL_VIDEODRIVER"] = "dummy";
        
        #pygame.mixer.init()
        pygame.init();
        self.clock = pygame.time.Clock();
        self.screen = pygame.display.set_mode((640,480));
        pygame.camera.init();
        self.cam = pygame.camera.Camera("/dev/video0",(640,480))

        self.cam.start()
        self.frames = collections.deque(maxlen=30); 
        #tts = gTTS(text='Go out now , we wa we wa', lang='en')
        #tts.save("theif1.mp3");
        #tts = gTTS(text='اخرج الان , وي وا وي وا', lang='ar')
        #tts.save("theif2.mp3");
        self.err = None;
        
        

        
    

    def run(self,args):

        print ('started');
        i = 0;
        while (True):
            pygame.event.pump();
            events = pygame.event.get()
            myEvent = None;
            for event in events:
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        myEvent = True;
                       
            if (myEvent == True):
                break;
            found5 = False;
            if(i >= 500):
                found5 = True;
                i = 0;
                
                #do get image and do work
                image = self.cam.get_image();
                
                self.screen.blit(image, (0,0));
                

            self.clock.tick(30);
            i+= 30
            
            
            pygame.display.flip()
            if(found5 == True):
                #do image work;
                self.frames.append(pygame.surfarray.array3d(self.screen));
                arr = np.array(list(self.frames));
                myLen = len(arr);
                if(myLen > 2):
                    #do your checks here;
                    y1 = arr[myLen-1].reshape((640,-1));
                    y2 = arr[myLen-2].reshape((640,-1));
                    err = mean_squared_error(y1,y2);
                    foundErr = False;
                    if(self.err != None):
                        foundErr = True;
                    if(self.err == None):
                        self.err = err;
                    else :
                        if(err > (self.err + (self.err * 0.10))):
                            print(err);
                            
                            if(not pygame.mixer.music.get_busy()):
                                #pygame.mixer.music.load(os.path.join(".","input","theif1.mp3"));
                                pygame.mixer.music.load(os.path.join(".", "input","theif2.mp3"));
                                pygame.mixer.music.play(0);
                                
                    if(foundErr == True):
                        self.err = None;        
                        

                    #print(arr[0].shape)
                    #y1 =  arr[0].reshape((640,-1));
                    #print(y1.shape);
