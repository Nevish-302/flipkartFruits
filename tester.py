import requests
from requests.exceptions import RequestException
import logging
import os
import time
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#investors

class ServerTester:
    def __init__(self, base_url, image_path, timeout):
        self.base_url = base_url
        self.image_path = image_path
        self.timeout = timeout
        self.session = requests.Session()

    def test_endpoints(self):
        results = {}

        # 1. Check image file
        results['image_check'] = self.check_image()

        image_path = self.image_path

        # 2. Test endpoints
        endpoints = {
            '/test': 10,  # 10 second timeout for test endpoint
            '/detect_fruit': 30  # 30 second timeout for detect_fruit endpoint
        }

        for endpoint, timeout in endpoints.items():
            results[endpoint] = self.test_endpoint(endpoint, image_path, timeout)

        return results

    def check_image(self):
        try:
            # Fetch the image from the provided URL
            image_url = self.image_path
            img_response = requests.get(image_url)

            if img_response.status_code != 200:
                return {"status": "failure", "error": f"Image not found: {image_url}"}

            # Open the image using PIL
            img = Image.open(BytesIO(img_response.content))

            return {
                "status": "success",
                "details": {
                    "format": img.format,
                    "size": img.size,
                    "mode": img.mode,
                    "file_size": len(img_response.content)  # File size in bytes
                }
            }
        except Exception as e:
            return {"status": "failure", "error": str(e)}

    def test_endpoint(self, endpoint, image_url, timeout):
        try:
            # Fetch the image from the provided URL
            img_response = requests.get(image_url, timeout=timeout)

            if img_response.status_code != 200:
                return {
                    "status": "failure",
                    "status_code": img_response.status_code,
                    "error": img_response.text
                }

            start_time = time.time()

            # Send the fetched image to the endpoint
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                files={'image': ('image.jpg', img_response.content)},  # Use a dummy filename
                timeout=timeout
            )

            elapsed_time = time.time() - start_time

            return {
                "status": "success" if response.status_code == 200 else "failure",
                "status_code": response.status_code,
                "elapsed_time": f"{elapsed_time:.2f} seconds",
                "response": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }

        except requests.RequestException as e:
            return {"status": "failure", "error": str(e)}


def print_results(results):
    print("\n=== Test Results ===")
    for test_name, result in results.items():
        print(f"\n{test_name}:")
        for key, value in result.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    BASE_URL = "http://127.0.0.1:5000"
    IMAGE_PATH = 'https://img.freepik.com/free-photo/picture-nice-red-apple-white-background_125540-4627.jpg'
    timeout = 10
    tester = ServerTester(BASE_URL, IMAGE_PATH, timeout)
    results = tester.test_endpoints()
    print_results(results)