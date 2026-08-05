from setuptools import find_packages, setup

package_name = 'sra_voice_input'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='master',
    maintainer_email='master@todo.todo',
    description='SRA voice input node using Faster-Whisper',
    license='MIT',
    entry_points={
        'console_scripts': [
            'voice_input = sra_voice_input.voice_input:main',
        ],
    },
)
