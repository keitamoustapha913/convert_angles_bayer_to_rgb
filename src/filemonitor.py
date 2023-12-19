import sys
import time
import logging
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

from utils import extract_img_polarized_angles_bayer_rg8_path


class FileMonitorEventHandler(LoggingEventHandler):
    """Logs all the events captured."""

    def __init__(self, *args, 
                      logger: Optional[logging.Logger] = None, 
                      unzipped_directory_path: str = "", 
                      **kwargs):
        super(FileMonitorEventHandler).__init__(*args, **kwargs)
        self.logger = logger
        self.unzipped_directory_path = unzipped_directory_path


    def on_moved(self, event) -> None:
        pass

    def on_any_event(self, event) -> None:
        pass

    def on_created(self, event) -> None:
        #super().on_created(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info("Created %s: %s", what, event.src_path)

    def on_deleted(self, event) -> None:
        pass

    def on_modified(self, event) -> None:
        pass

    def on_closed(self, event) -> None:

        what = "directory" if event.is_directory else "file"
        self.logger.info("Closed %s: %s", what, event.src_path)
        extract_img_polarized_angles_bayer_rg8_path(zipped_compressed_imgs_path = event.src_path, unzipped_compressed_dir = self.unzipped_directory_path )
        

    def on_opened(self, event) -> None:
        pass



class FileMonitor():
    def __init__(self , *args, watch_directory_path:str = "", unzipped_directory_path:str = "" , **kwargs):
        self.watch_directory_path = watch_directory_path
        self.unzipped_directory_path = unzipped_directory_path
        self.create_logger()

    def create_logger(self):
        """Create and configure logger"""
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
        
        # Creating an object
        self.logger = logging.getLogger()

    def start(self):
        # Initialize logging event handler
        event_handler = FileMonitorEventHandler(logger = self.logger , unzipped_directory_path = self.unzipped_directory_path)
        print(f"self.watch_directory_path : {self.watch_directory_path}")
        # Initialize the Observer
        observer = Observer()
        observer.schedule(event_handler, self.watch_directory_path, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)
        finally:
            observer.stop()
            observer.join()

    def stop(self):
        pass



    
if __name__ == "__main__":
    watch_directory_path = "/home/theyeq-admin/Documents/convert_angles_bayer_to_rgb/images/zipped"
    unzipped_directory_path = "/home/theyeq-admin/Documents/convert_angles_bayer_to_rgb/images/unzipped"
    file_monitor = FileMonitor(watch_directory_path = watch_directory_path, unzipped_directory_path = unzipped_directory_path)
    file_monitor.start()