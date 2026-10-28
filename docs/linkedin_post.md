About 1.1 million people in Pakistan are blind, and most payments are still cash. For them, every banknote is a question: is this 500 or 5,000? The usual answer is to ask a stranger and trust them.

For my Deep Learning semester project I built BoltaNote (بولتا نوٹ, "the talking note"): point your phone at a Pakistani note and it says the value out loud, in Urdu or English. It's a web page, so there's nothing to install, and it works offline once loaded.

What I learned building it:
• Check your test data first: 339 of 402 test photos in a second public currency dataset were near-copies of the first. Only 63 were genuinely new.
• Near-perfect accuracy on the dataset's own photos (100%) means little. On photos by other people it scored 92%.
• Silence beats a wrong answer: it only speaks above a confidence threshold calibrated on stress-tested photos, and when 3 frames in a row agree.
• MobileNet (ImageNet-pretrained) generalised far better than a CNN trained from scratch, and runs fast enough for a cheap phone.

Try it on your phone: https://boltanote.vercel.app
Code, weights and my own test photos: https://github.com/Hassan-Ali-Mehdi-3024/boltanote
Walkthrough video: https://youtu.be/Je7tUmXsgO8

Thanks to Dr. Munir Ahmad, and to the authors of the open Pakistani currency datasets.

#DeepLearning #Accessibility #ComputerVision #AssistiveTechnology #Pakistan #SuperiorUniversity
