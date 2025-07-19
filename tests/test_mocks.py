"""
Comprehensive mock infrastructure for PCB inspection system testing.

This module provides mock objects for all external dependencies to enable
testing without requiring actual hardware or complex library installations.
"""

import sys
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Any, Dict, List, Optional


class MockCV2:
    """Comprehensive OpenCV mock for testing."""
    
    # Constants
    NORM_MINMAX = 32
    CV_64F = 6
    COLOR_BAYER_RG2GRAY = 46
    COLOR_GRAY2BGR = 8
    COLOR_GRAY2RGB = 8
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 2
    MORPH_RECT = 0
    MORPH_CLOSE = 3
    MORPH_OPEN = 2
    FONT_HERSHEY_SIMPLEX = 0
    INTER_LINEAR = 1
    
    @staticmethod
    def normalize(src, dst=None, alpha=0, beta=255, norm_type=32):
        """Mock cv2.normalize."""
        return src if dst is None else dst
    
    @staticmethod
    def createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)):
        """Mock CLAHE object."""
        clahe = Mock()
        clahe.apply = lambda img: img
        return clahe
    
    @staticmethod
    def bilateralFilter(src, d, sigmaColor, sigmaSpace):
        """Mock bilateral filter."""
        return src
    
    @staticmethod
    def cvtColor(src, code):
        """Mock color conversion."""
        if code == MockCV2.COLOR_GRAY2BGR or code == MockCV2.COLOR_GRAY2RGB:
            if len(src.shape) == 2:
                return np.stack([src, src, src], axis=2)
        elif code == MockCV2.COLOR_BAYER_RG2GRAY:
            return src[:, :, 0] if len(src.shape) > 2 else src
        return src
    
    @staticmethod
    def resize(src, dsize, fx=0, fy=0, interpolation=1):
        """Mock image resize."""
        if len(src.shape) == 2:
            return np.random.randint(0, 255, dsize[::-1], dtype=np.uint8)
        else:
            return np.random.randint(0, 255, (*dsize[::-1], src.shape[2]), dtype=np.uint8)
    
    @staticmethod
    def Canny(image, threshold1, threshold2):
        """Mock Canny edge detection."""
        return np.random.randint(0, 255, image.shape, dtype=np.uint8)
    
    @staticmethod
    def findContours(image, mode, method):
        """Mock contour detection."""
        contour = np.array([[[50, 50]], [[100, 50]], [[100, 100]], [[50, 100]]])
        return [contour], None
    
    @staticmethod
    def boundingRect(contour):
        """Mock bounding rectangle."""
        return (40, 40, 60, 60)
    
    @staticmethod
    def contourArea(contour):
        """Mock contour area."""
        return 3600
    
    @staticmethod
    def Laplacian(src, ddepth):
        """Mock Laplacian operation."""
        return np.random.randn(*src.shape) * 50
    
    @staticmethod
    def getStructuringElement(shape, ksize):
        """Mock morphological structuring element."""
        return np.ones(ksize, dtype=np.uint8)
    
    @staticmethod
    def morphologyEx(src, op, kernel):
        """Mock morphological operation."""
        return src
    
    @staticmethod
    def dilate(src, kernel, iterations=1):
        """Mock dilation."""
        return src
    
    @staticmethod
    def erode(src, kernel, iterations=1):
        """Mock erosion."""
        return src
    
    @staticmethod
    def GaussianBlur(src, ksize, sigmaX):
        """Mock Gaussian blur."""
        return src
    
    @staticmethod
    def rectangle(img, pt1, pt2, color, thickness=1):
        """Mock rectangle drawing."""
        pass
    
    @staticmethod
    def putText(img, text, org, fontFace, fontScale, color, thickness=1):
        """Mock text drawing."""
        pass
    
    @staticmethod
    def getTextSize(text, fontFace, fontScale, thickness):
        """Mock text size calculation."""
        return ((100, 20), 5)
    
    @staticmethod
    def arcLength(curve, closed):
        """Mock arc length calculation."""
        return 240.0
    
    @staticmethod
    def approxPolyDP(curve, epsilon, closed):
        """Mock polygon approximation."""
        return curve
    
    @staticmethod
    def imwrite(filename, img):
        """Mock image writing."""
        return True


class MockTorch:
    """Mock PyTorch for AI testing."""
    
    class cuda:
        @staticmethod
        def is_available():
            return True
        
        @staticmethod
        def device_count():
            return 1
        
        @staticmethod
        def empty_cache():
            pass
        
        @staticmethod
        def get_device_name(device_id=0):
            return 'Mock GPU Device'
        
        @staticmethod
        def get_device_properties(device_id=0):
            class MockProps:
                total_memory = 8 * 1024 * 1024 * 1024  # 8GB
            return MockProps()


class MockUltralytics:
    """Mock Ultralytics YOLO for AI testing."""
    
    class YOLO:
        def __init__(self, model_path):
            self.model_path = model_path
        
        def to(self, device):
            return self
        
        def half(self):
            return self
        
        def __call__(self, image, **kwargs):
            # Return mock results
            result = Mock()
            result.boxes = None
            return [result]


class MockTkinter:
    """Mock tkinter for GUI testing."""
    
    class Widget:
        def __init__(self, parent=None, **kwargs):
            pass
        def pack(self, **kwargs): pass
        def grid(self, **kwargs): pass
        def place(self, **kwargs): pass
        def config(self, **kwargs): pass
        def configure(self, **kwargs): pass
        def bind(self, event, callback): pass
        def focus_set(self): pass
        def destroy(self): pass
    
    class Tk(Widget):
        def title(self, title): pass
        def geometry(self, geom): pass
        def minsize(self, width, height): pass
        def maxsize(self, width, height): pass
        def resizable(self, width, height): pass
        def mainloop(self): pass
        def grid_columnconfigure(self, col, weight): pass
        def grid_rowconfigure(self, row, weight): pass
        def columnconfigure(self, col, weight): pass
        def rowconfigure(self, row, weight): pass
        def after(self, delay, callback): pass
        def protocol(self, protocol, callback): pass
        def withdraw(self): pass
        def deiconify(self): pass
        def lift(self): pass
        def attributes(self, *args): pass
    
    Frame = Widget
    Label = Widget
    Button = Widget
    Text = Widget
    Canvas = Widget
    Scrollbar = Widget
    Entry = Widget
    Scale = Widget
    Checkbutton = Widget
    Radiobutton = Widget
    Listbox = Widget
    
    class ttk:
        Frame = MockTkinter.Widget
        Label = MockTkinter.Widget
        Button = MockTkinter.Widget
        LabelFrame = MockTkinter.Widget
        Progressbar = MockTkinter.Widget
        Combobox = MockTkinter.Widget
        Treeview = MockTkinter.Widget
        Separator = MockTkinter.Widget
        Notebook = MockTkinter.Widget
    
    class messagebox:
        @staticmethod
        def showinfo(title, message): pass
        @staticmethod
        def showerror(title, message): pass
        @staticmethod
        def askquestion(title, message): return 'yes'
        @staticmethod
        def askyesno(title, message): return True
    
    class filedialog:
        @staticmethod
        def asksaveasfilename(**kwargs): return 'test_file.json'
        @staticmethod
        def askopenfilename(**kwargs): return 'test_file.json'
    
    # Constants
    TOP = 'top'
    BOTTOM = 'bottom'
    LEFT = 'left'
    RIGHT = 'right'
    BOTH = 'both'
    X = 'x'
    Y = 'y'
    W = 'w'
    E = 'e'
    N = 'n'
    S = 's'
    NW = 'nw'
    NE = 'ne'
    SW = 'sw'
    SE = 'se'
    END = 'end'
    DISABLED = 'disabled'
    NORMAL = 'normal'
    ACTIVE = 'active'
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'


class MockPIL:
    """Mock PIL for image processing."""
    
    class Image:
        @staticmethod
        def fromarray(arr):
            return MockPIL.Image()
        
        def thumbnail(self, size):
            pass
        
        def resize(self, size):
            return self
    
    class ImageTk:
        @staticmethod
        def PhotoImage(image):
            return 'mock_photo'


class MockMatplotlib:
    """Mock matplotlib for plotting."""
    
    class pyplot:
        @staticmethod
        def figure(**kwargs):
            return 'mock_figure'
        
        @staticmethod
        def subplots(**kwargs):
            return ('mock_fig', 'mock_ax')
        
        @staticmethod
        def close(fig):
            pass
        
        @staticmethod
        def show():
            pass
    
    class backends:
        class backend_tkagg:
            class FigureCanvasTkAgg:
                def __init__(self, figure, parent):
                    pass
                
                def get_tk_widget(self):
                    return MockTkinter.Widget()
                
                def draw(self):
                    pass


class MockPypylon:
    """Mock pypylon for camera testing."""
    
    class pylon:
        class InstantCamera:
            def __init__(self, device=None):
                pass
            
            def Open(self):
                pass
            
            def Close(self):
                pass
            
            def StartGrabbing(self, strategy=None):
                pass
            
            def StopGrabbing(self):
                pass
            
            def GrabOne(self, timeout):
                result = Mock()
                result.GrabSucceeded.return_value = True
                result.Array = np.random.randint(0, 255, (1024, 768), dtype=np.uint8)
                result.Release = Mock()
                return result
        
        class TlFactory:
            @staticmethod
            def GetInstance():
                factory = Mock()
                factory.CreateFirstDevice.return_value = Mock()
                return factory
        
        RegistrationMode_Append = 'append'
        Cleanup_Delete = 'delete'
        GrabStrategy_LatestImageOnly = 'latest'


def setup_test_environment():
    """Setup complete mock environment for testing."""
    
    # Mock all external dependencies
    sys.modules['cv2'] = MockCV2()
    sys.modules['torch'] = MockTorch()
    sys.modules['ultralytics'] = MockUltralytics()
    sys.modules['tkinter'] = MockTkinter()
    sys.modules['tkinter.ttk'] = MockTkinter.ttk()
    sys.modules['tkinter.messagebox'] = MockTkinter.messagebox()
    sys.modules['tkinter.filedialog'] = MockTkinter.filedialog()
    sys.modules['PIL'] = MockPIL()
    sys.modules['PIL.Image'] = MockPIL.Image()
    sys.modules['PIL.ImageTk'] = MockPIL.ImageTk()
    sys.modules['matplotlib'] = MockMatplotlib()
    sys.modules['matplotlib.pyplot'] = MockMatplotlib.pyplot()
    sys.modules['matplotlib.backends'] = MockMatplotlib.backends()
    sys.modules['matplotlib.backends.backend_tkagg'] = MockMatplotlib.backends.backend_tkagg()
    sys.modules['pypylon'] = MockPypylon()
    
    print("✅ Test environment setup complete with comprehensive mocks")


def teardown_test_environment():
    """Clean up mock environment."""
    mock_modules = [
        'cv2', 'torch', 'ultralytics', 'tkinter', 'tkinter.ttk',
        'tkinter.messagebox', 'tkinter.filedialog', 'PIL', 'PIL.Image',
        'PIL.ImageTk', 'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends',
        'matplotlib.backends.backend_tkagg', 'pypylon'
    ]
    
    for module in mock_modules:
        if module in sys.modules:
            del sys.modules[module]
    
    print("✅ Test environment cleaned up")


if __name__ == "__main__":
    # Test the mock environment
    setup_test_environment()
    
    # Test imports work
    try:
        import cv2
        import torch
        import tkinter
        print("✅ All mock imports successful")
    except ImportError as e:
        print(f"❌ Mock import failed: {e}")
    
    teardown_test_environment()