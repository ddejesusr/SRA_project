from setuptools import setup

package_name = 'sra_tts'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='SRA TTS node — Piper offline speech synthesis',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'tts_node = sra_tts.tts_node:main',
        ],
    },
)
