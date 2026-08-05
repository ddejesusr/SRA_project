import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sra_web'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install the static folder so FastAPI can serve index.html
        (os.path.join('share', package_name, 'static'),
            glob('static/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='master',
    maintainer_email='master@todo.todo',
    description='SRA web dashboard — FastAPI + WebSocket bridge',
    license='MIT',
    entry_points={
        'console_scripts': [
            'web_bridge = sra_web.web_bridge:main',
        ],
    },
)
