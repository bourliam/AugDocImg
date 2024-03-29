import sys
import os
import time
import tarfile

import urllib.request as urlrequest
import urllib.parse as urlparse


class ImageNetDownloader:
    def __init__(self):
        self.host = 'http://www.image-net.org'

    def download_file(self, url, desc=None, renamed_file=None):
        u = urlrequest.urlopen(url)

        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        filename = os.path.basename(path)
        if not filename:
            filename = 'downloaded.file'

        if not renamed_file is None:
            filename = renamed_file

        if desc:
            filename = os.path.join(desc, filename)

        with open(filename, 'wb') as f:
            meta = u.info()
            meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
            meta_length = meta_func("Content-Length")
            file_size = None
            if meta_length:
                file_size = int(meta_length[0])
            print("Downloading: {0} Bytes: {1}".format(url, file_size))

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)

                status = "{0:16}".format(file_size_dl)
                if file_size:
                    status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
                status += chr(13)

        return filename


    def mkDir(self, title):
        if not os.path.exists(title):
            os.mkdir(title)
        return os.path.abspath(title)

    def downloadImagesByURLs(self, title, imageUrls, number = 5):
        # save to the dir e.g: dog_images/
        imgdir = os.path.join(self.mkDir("images"), str(title))
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)

        count = 0
        for url in imageUrls:
            if(count == number):
                break
            try:
                self.download_file(url, imgdir)
                count += 1
            except (Exception):
                print('Failed to download : ' + url)
