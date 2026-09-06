import os
from verifier import get_embedding, verify_speaker

if __name__ == "__main__":
    enrolled = get_embedding("./data/LA/LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac")
    print("Same person:", verify_speaker("./data/LA/LA/ASVspoof2019_LA_train/flac/LA_T_1271820.flac", enrolled))
    print("Different person:", verify_speaker("./data/LA/LA/ASVspoof2019_LA_train/flac/LA_T_1078395.flac", enrolled))
