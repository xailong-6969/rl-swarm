from abc import ABC, abstractmethod
class DatasetMapper(ABC):
    """Base class for dataset-specific mappers.
    
    Each dataset mapper is responsible for transforming dataset-specific fields
    into the standard format required by the system.
    """
    
    @abstractmethod
    def map_prompt(self, batch: dict, index: int) -> str:
        """Extract and format the prompt for a given index.
        
        Args:
            batch: The batch data containing dataset-specific fields
            index: The index of the item to extract
            
        Returns:
            The formatted prompt string
        """
        raise NotImplementedError
    
    @abstractmethod
    def map_test(self, batch: dict, index: int) -> str:
        """Extract and format the test for a given index.
        
        Args:
            batch: The batch data containing dataset-specific fields
            index: The index of the item to extract
            
        Returns:
            The formatted test string
        """
        raise NotImplementedError
    
    @abstractmethod
    def format_question(self, prompt: str, test: str) -> str:
        """Format the final question combining prompt and test.
        
        Args:
            prompt: The base prompt
            test: The test string
            
        Returns:
            The formatted question to be used in the environment state
        """
        raise NotImplementedError


class MBPPMapper(DatasetMapper):
    """Mapper for MBPP (Mostly Basic Programming Problems) dataset."""
    
    def map_prompt(self, batch: dict, index: int) -> str:
        return batch.get('text', [])[index]
    
    def map_test(self, batch: dict, index: int) -> str:
        test_imports = batch.get('test_setup_code', [])[index]
        test_list = batch.get('test_list', [])[index]
        return test_imports + "\n" + "\n".join(test_list)
    
    def format_question(self, prompt: str, test: str) -> str:
        # MBPP needs function name hints from tests
        return prompt + '\nplease match the function name to the following test\n' + test


class CodeContestsMapper(DatasetMapper):
    """Mapper for Code Contests dataset."""
    
    def map_prompt(self, batch: dict, index: int) -> str:
        return batch.get('description', [])[index]
    
    def map_test(self, batch: dict, index: int) -> str:
        test_dict = batch.get('public_tests', [])[index]
        return str(test_dict)
    
    def format_question(self, prompt: str, test: str) -> str:
        # Code Contests uses prompt as-is
        return prompt
