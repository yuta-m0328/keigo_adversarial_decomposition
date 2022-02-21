import glob
import os

def remove_glob(pathname):
    for p in pathname:
        if os.path.isfile(p):
            os.remove(p)

def main():
    pathname = glob.glob('./*.png')
    print(pathname)
    remove_glob(pathname)

if __name__ == '__main__':
    main()
