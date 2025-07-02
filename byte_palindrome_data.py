import torch
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from data import cycle_loader
import string
import re


class BytePalindromeProcessor:
    """Byte-level processor for palindrome text"""
    
    def __init__(self, max_length=128):
        self.max_length = max_length
        # Byte vocabulary: 0-255 for all possible bytes, plus special tokens
        self.vocab_size = 256 + 3  # 256 bytes + PAD + BOS + EOS
        self.PAD = 256
        self.BOS = 257  # Beginning of sequence
        self.EOS = 258  # End of sequence
        
    def text_to_bytes(self, text):
        """Convert text to byte sequence"""
        return list(text.encode('utf-8'))
    
    def bytes_to_text(self, byte_seq):
        """Convert byte sequence back to text"""
        try:
            # Filter out special tokens
            clean_bytes = [b for b in byte_seq if b < 256]
            return bytes(clean_bytes).decode('utf-8', errors='ignore')
        except:
            return ""
    
    def encode_palindrome(self, text):
        """Encode palindrome text to byte sequence with special tokens"""
        byte_seq = self.text_to_bytes(text)
        
        # Add BOS and EOS tokens
        encoded = [self.BOS] + byte_seq + [self.EOS]
        
        # Pad or truncate to max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length-1] + [self.EOS]
        else:
            encoded = encoded + [self.PAD] * (self.max_length - len(encoded))
            
        return encoded
    
    def decode_palindrome(self, byte_seq):
        """Decode byte sequence back to text"""
        # Remove padding and special tokens for text reconstruction
        clean_seq = []
        for b in byte_seq:
            if b == self.EOS:
                break
            elif b != self.PAD and b != self.BOS and b < 256:
                clean_seq.append(b)
        
        return self.bytes_to_text(clean_seq)


def is_palindrome(text):
    """Check if text is a palindrome (ignoring spaces, punctuation, case)"""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
    return cleaned == cleaned[::-1] and len(cleaned) > 0


def generate_character_palindromes(length_range=(3, 30), count=1000):
    """Generate random character-level palindromes"""
    palindromes = []
    # Use printable ASCII characters
    chars = string.ascii_letters + string.digits + ' .,!?'
    
    for _ in range(count):
        # Random length
        length = np.random.randint(length_range[0], length_range[1] + 1)
        
        if length % 2 == 0:
            # Even length: create two halves
            half_length = length // 2
            first_half = ''.join(np.random.choice(list(chars), half_length))
            palindrome = first_half + first_half[::-1]
        else:
            # Odd length: create two halves with middle character
            half_length = length // 2
            first_half = ''.join(np.random.choice(list(chars), half_length))
            middle = np.random.choice(list(chars))
            palindrome = first_half + middle + first_half[::-1]
        
        # Clean up the palindrome
        palindrome = palindrome.strip()
        if len(palindrome) >= 3:
            palindromes.append(palindrome)
    
    return palindromes


def create_word_palindromes():
    """Create word-based palindromes"""
    # Single word palindromes
    word_palindromes = [
        "level", "radar", "civic", "refer", "stats", "noon", "deed", "peep",
        "mom", "dad", "pop", "wow", "bob", "nun", "eye", "gag", "pip", "sis",
        "tot", "tut", "kayak", "rotor", "madam", "racecar", "redder", "solos"
    ]
    
    # Famous phrase palindromes
    phrase_palindromes = [
        "A man a plan a canal Panama",
        "Madam Im Adam",
        "Was it a car or a cat I saw",
        "Never odd or even",
        "Do geese see God",
        "Step on no pets",
        "Yo banana boy",
        "Was it a rat I saw",
        "Able was I ere I saw Elba",
        "Mr Owl ate my metal worm",
        "No lemon no melon",
        "Taco cat",
        "Race car",
        "A Santa at NASA",
        "Dammit Im mad",
        "Evil is a name of a foeman as I live"
    ]
    
    return word_palindromes + phrase_palindromes


def create_byte_palindrome_dataset(size=10000, max_length=128):
    """Create a dataset of palindromes for byte-level processing"""
    
    palindromes = []
    
    # Add word-based palindromes
    word_palindromes = create_word_palindromes()
    palindromes.extend(word_palindromes)
    
    # Generate character-level palindromes
    char_palindromes = generate_character_palindromes(count=size - len(word_palindromes))
    palindromes.extend(char_palindromes)
    
    # Ensure we have enough palindromes
    while len(palindromes) < size:
        additional = generate_character_palindromes(count=size - len(palindromes))
        palindromes.extend(additional)
    
    # Shuffle and trim to exact size
    np.random.shuffle(palindromes)
    palindromes = palindromes[:size]
    
    # Verify all are palindromes and filter by length
    processor = BytePalindromeProcessor(max_length)
    valid_palindromes = []
    
    for p in palindromes:
        if is_palindrome(p) and len(processor.text_to_bytes(p)) + 2 <= max_length:  # +2 for BOS/EOS
            valid_palindromes.append(p)
    
    return valid_palindromes


def get_byte_palindrome_dataset(mode="train", cache_dir=None, block_size=128, num_proc=8, dataset_size=10000):
    """Get byte-level palindrome dataset"""
    
    palindromes = create_byte_palindrome_dataset(size=dataset_size, max_length=block_size)
    processor = BytePalindromeProcessor(max_length=block_size)
    
    # Encode palindromes to byte sequences
    encoded_data = []
    for palindrome in palindromes:
        byte_seq = processor.encode_palindrome(palindrome)
        encoded_data.append({"input_ids": byte_seq})
    
    dataset = Dataset.from_list(encoded_data)
    dataset = dataset.with_format('torch')
    
    return dataset


def get_byte_palindrome_dataloaders(config, distributed=True):
    """Get byte-level palindrome dataloaders"""
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    train_set = get_byte_palindrome_dataset("train", cache_dir=config.data.cache_dir, block_size=config.model.length, dataset_size=10000)
    valid_set = get_byte_palindrome_dataset("validation", cache_dir=config.data.cache_dir, block_size=config.model.length, dataset_size=1000)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader


if __name__ == "__main__":
    # Test the byte-level palindrome processing
    processor = BytePalindromeProcessor(max_length=64)
    
    # Test palindromes
    test_palindromes = ["racecar", "A man a plan a canal Panama", "level", "hello world"]
    
    print("Testing byte-level palindrome processing:")
    for palindrome in test_palindromes:
        encoded = processor.encode_palindrome(palindrome)
        decoded = processor.decode_palindrome(encoded)
        is_pal = is_palindrome(palindrome)
        
        print(f"Original: '{palindrome}' (palindrome: {is_pal})")
        print(f"Encoded length: {len([x for x in encoded if x != processor.PAD])}")
        print(f"Decoded: '{decoded}'")
        print(f"Match: {palindrome.lower() == decoded.lower()}")
        print("-" * 50)
    
    # Test dataset creation
    print("\nTesting dataset creation:")
    dataset = get_byte_palindrome_dataset(dataset_size=10)
    print(f"Created dataset with {len(dataset)} samples")
    
    # Show first few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]['input_ids']
        decoded = processor.decode_palindrome(sample.tolist())
        print(f"Sample {i+1}: '{decoded}' (palindrome: {is_palindrome(decoded)})")