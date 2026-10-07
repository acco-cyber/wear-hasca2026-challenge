The **3rd WEAR Dataset Challenge** is a Human Activity Recognition prediction challenge based on the [WEAR dataset](http://mariusbock.github.io/wear/). The challenge will be part of the [HASCA Workshop](http://hasca2026.hasc.jp/) at [UbiComp/ ISWC 2026](https://www.ubicomp.org/ubicomp-iswc-2026/).

With previous iterations of the WEAR challenge focusing solely on inertial data, this year we want to challenge the wearable community to explore how to most effectively combine egocentric cameras with inertial sensors. 
To do so, this year, we provide random 1-second sliding windows from a single inertial sensor, as well as pre-extracted, frame-wise features ([VideoMAEv2-Base](https://huggingface.co/OpenGVLab/VideoMAEv2-Base)) from the egocentric camera's video stream.

⚠️ **Update – April 26, 2025:** A bug in the video feature generation has been fixed. Please redownload the test data. Sample ordering is unchanged; prior submissions without video features are unaffected.