// Compile via:
//   gcc -shared -O3 -Wall -fPIC -lX11 -Wl,-soname,screenshot -o screenshot.so screenshot.c -lX11
// code from http://stackoverflow.com/questions/69645/take-a-screenshot-via-a-python-script-linux
#include <stdio.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

void getScreen(const int, const int, const int, const int, unsigned char *);
void getScreen(const int xx, const int yy, const int W, const int H, /*out*/ unsigned char * data) 
{
   Display *display = XOpenDisplay(NULL);
   Window root = DefaultRootWindow(display);

   XImage *image = XGetImage(display, root, xx, yy, W, H, AllPlanes, ZPixmap);

   unsigned long red_mask   = image->red_mask;
   unsigned long green_mask = image->green_mask;
   unsigned long blue_mask  = image->blue_mask;
   int x, y;
   int ii = 0;
   for (y = 0; y < H; y++) {
       for (x = 0; x < W; x++) {
         unsigned long pixel = XGetPixel(image, x, y);
         unsigned char blue  = (pixel & blue_mask);
         unsigned char green = (pixel & green_mask) >> 8;
         unsigned char red   = (pixel & red_mask) >> 16;

         data[ii + 2] = blue;
         data[ii + 1] = green;
         data[ii + 0] = red;
         ii += 3;
      }
   }
   XDestroyImage(image);
   XDestroyWindow(display, root);
   XCloseDisplay(display);
}
