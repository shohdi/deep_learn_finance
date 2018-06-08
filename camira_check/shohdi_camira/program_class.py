
import pygame;
import pygame.mixer;
import pygame.camera;
import os;
import collections;
import numpy as np ;
from sklearn.metrics import mean_squared_error
from gtts import gTTS
import uuid;
import smtplib

# Here are the email package modules we'll need
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart







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
        #tts.save(os.path.join(".", "input","theif1.mp3"));
        #tts = gTTS(text='اخرج الان , وي وا وي وا', lang='ar')
        #tts.save(os.path.join(".", "input","theif2.mp3"));
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
            if(i >= 250):
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
                    y1 = y1/255.0
                    y2 = y2/255.0
                    y1 = np.array(y1,dtype='float32');
                    y2 = np.array(y2,dtype='float32');
                    err = mean_squared_error(y1,y2);
                    foundErr = False;
                    if(self.err != None):
                        foundErr = True;
                    if(self.err == None):
                        self.err = err;
                    else :
                        if(err > (self.err + (self.err * 0.10))):
                            print(err);
                            import scipy.misc as smp
                            img1 = smp.toimage( arr[myLen-1]);
                            img2 = smp.toimage( arr[myLen-2])
                            myUuid = str(uuid.uuid4());
                            img1Name = os.path.join('output','img1_{0}.png'.format(myUuid));
                            img2Name = os.path.join('output','img2_{0}.png'.format(myUuid))
                            smp.imsave(img1Name ,img1);
                            smp.imsave(img2Name ,img2);

                            # Create the container (outer) email message.
                            msg = MIMEMultipart()
                            msg['Subject'] = 'Home Security Risk'
                            # me == the sender's email address
                            # family = the list of all recipients' email addresses
                            msg['From'] = 'shohdi_home@gmail.com'
                            msg['To'] = 'shohdi@gmail.com'
                            msg.preamble = 'Home Security Risk'

                            # Assume we know that the image files are all in PNG format

                            # Open the files in binary mode.  Let the MIMEImage class automatically
                            # guess the specific image type.
                            fp = open(img1Name, 'rb')
                            img1f = MIMEImage(fp.read())
                            fp.close()
                            msg.attach(img1f)
                            fp = open(img2Name, 'rb')
                            img2f = MIMEImage(fp.read())
                            fp.close()
                            msg.attach(img2f)

                            # Send the email via our own SMTP server.
                            s = smtplib.SMTP('')
                            s.sendmail('', [''], msg.as_string())
                            s.quit()



                            #if(not pygame.mixer.music.get_busy()):
                                #pygame.mixer.music.load(os.path.join(".","input","theif1.mp3"));
                                #pygame.mixer.music.load(os.path.join(".", "input","theif2.mp3"));
                                #pygame.mixer.music.play(0);

                                
                    if(foundErr == True):
                        self.err = None;        
                        

                    #print(arr[0].shape)
                    #y1 =  arr[0].reshape((640,-1));
                    #print(y1.shape);
