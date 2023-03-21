
import unittest
from filter import filter_item

class TestFilterItem(unittest.TestCase):
    def test_filter_item(self):
        # Test case 1
        metadata1 = {
            "genre": "action",
            "year": 2020,
            "length_hrs": 1.5
        }
        filter1 = {"genre": "action"}
        self.assertTrue(filter_item(filter1, metadata1))

        # Test case 2
        metadata2 = {
            "color": "blue",
            "fit": "straight",
            "price": 29.99,
            "is_jeans": True
        }
        filter2 = {"color": "blue", "is_jeans": True}
        self.assertTrue(filter_item(filter2, metadata2))

        # Test case 3
        metadata3 = {"genre": ["comedy", "documentary"]}
        filter3 = {"genre": "comedy"}
        self.assertTrue(filter_item(filter3, metadata3))

        # Test case 4
        filter4 = {"genre": {"$in": ["documentary", "action"]}}
        self.assertTrue(filter_item(filter4, metadata3))

        # Test case 5
        filter5 = {"$and": [{"genre": "comedy"}, {"genre": "documentary"}]}
        self.assertTrue(filter_item(filter5, metadata3))

        # Test case 6
        filter6 = {"$and": [{"genre": "comedy"}, {"genre": "drama"}]}
        self.assertFalse(filter_item(filter6, metadata3))

        # Test case 7
        filter7 = {"$or": [{"genre": "comedy"}, {"genre": "drama"}]}
        self.assertTrue(filter_item(filter7, metadata3))

        # Test case 8
        filter8 = {"$or": [{"genre": "action"}, {"genre": "drama"}]}
        self.assertFalse(filter_item(filter8, metadata3))

        # Test case 9
        metadata4 = {
            "genre": ["action", "comedy"],
            "year": 2018,
            "rating": 7.5,
            "is_popular": True
        }
        filter9 = {
            "$and": [
                {"genre": {"$in": ["action", "drama"]}},
                {"year": {"$gte": 2010, "$lte": 2020}},
                {"rating": {"$gt": 7}},
                {"is_popular": True}
            ]
        }
        self.assertTrue(filter_item(filter9, metadata4))

        # Test case 10
        filter10 = {
            "$or": [
                {"genre": {"$in": ["drama", "thriller"]}},
                {"year": {"$lt": 2000}},
                {"rating": {"$lte": 5}},
                {"is_popular": False}
            ]
        }
        self.assertFalse(filter_item(filter10, metadata4))

        # Test case 11
        metadata5 = {
            "genre": ["sci-fi", "adventure"],
            "year": 2021,
            "rating": 8.5,
            "is_popular": False
        }
        filter11 = {
            "$and": [
                {"genre": {"$nin": ["action", "drama"]}},
                {"year": {"$gte": 2010, "$lte": 2020}},
                {"rating": {"$gt": 7}},
                {"is_popular": False}
            ]
        }
        self.assertFalse(filter_item(filter11, metadata5))

        # Test case 12
        filter12 = {
            "$and": [
                {"genre": {"$nin": ["action", "drama"]}},
                {"year": {"$gte": 2010, "$lte": 2021}},
                {"rating": {"$gt": 7}},
                {"is_popular": False}
            ]
        }
        self.assertTrue(filter_item(filter12, metadata5))

        # Test case 13
        metadata6 = {
            "genre": ["horror", "thriller"],
            "year": 1999,
            "rating": 6.5,
            "is_popular": True
        }
        filter13 = {
            "$or": [
                {"genre": {"$in": ["horror", "thriller"]}},
                {"year": {"$lt": 2000}},
                {"rating": {"$lte": 5}},
                {"is_popular": False}
            ]
        }
        self.assertTrue(filter_item(filter13, metadata6))

class TestFilterItem2(unittest.TestCase):

    def test_eq_string(self):
        filter_dict = {"genre": {"$eq": "action"}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_eq_number(self):
        filter_dict = {"year": {"$eq": 2020}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_eq_boolean(self):
        filter_dict = {"is_jeans": {"$eq": True}}
        metadata_dict = {"color": "blue", "fit": "straight", "price": 29.99, "is_jeans": True}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_ne_string(self):
        filter_dict = {"genre": {"$ne": "comedy"}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_ne_number(self):
        filter_dict = {"year": {"$ne": 2019}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_ne_boolean(self):
        filter_dict = {"is_jeans": {"$ne": False}}
        metadata_dict = {"color": "blue", "fit": "straight", "price": 29.99, "is_jeans": True}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_gt(self):
        filter_dict = {"year": {"$gt": 2019}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_gte(self):
        filter_dict = {"year": {"$gte": 2020}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_lt(self):
        filter_dict = {"year": {"$lt": 2021}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_lte(self):
        filter_dict = {"year": {"$lte": 2020}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_in_string(self):
        filter_dict = {"genre": {"$in": ["action", "comedy"]}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_in_number(self):
        filter_dict = {"year": {"$in": [2019, 2020]}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_nin_string(self):
        filter_dict = {"genre": {"$nin": ["comedy", "drama"]}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_nin_number(self):
        filter_dict = {"year": {"$nin": [2019, 2021]}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_and(self):
        filter_dict = {"$and": [{"genre": "action"}, {"year": 2020}]}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_or(self):
        filter_dict = {"$or": [{"genre": "action"}, {"year": 2019}]}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_and_or(self):
        filter_dict = {"$and": [{"genre": "action"}, {"$or": [{"year": 2020}, {"length_hrs": 2}]}]}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_list_metadata(self):
        filter_dict = {"genre": "comedy"}
        metadata_dict = {"genre": ["comedy", "documentary"]}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_in_list_metadata(self):
        filter_dict = {"genre": {"$in": ["documentary", "action"]}}
        metadata_dict = {"genre": ["comedy", "documentary"]}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_and_list_metadata(self):
        filter_dict = {"$and": [{"genre": "comedy"}, {"genre": "documentary"}]}
        metadata_dict = {"genre": ["comedy", "documentary"]}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_eq_string_case_sensitive(self):
        filter_dict = {"genre": {"$eq": "Action"}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertFalse(filter_item(filter_dict, metadata_dict))

    def test_ne_string_case_sensitive(self):
        filter_dict = {"genre": {"$ne": "Action"}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_in_string_case_sensitive(self):
        filter_dict = {"genre": {"$in": ["Action", "Comedy"]}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertFalse(filter_item(filter_dict, metadata_dict))

    def test_nin_string_case_sensitive(self):
        filter_dict = {"genre": {"$nin": ["Action", "Drama"]}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_and_not_matching(self):
        filter_dict = {"$and": [{"genre": "action"}, {"year": 2019}]}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertFalse(filter_item(filter_dict, metadata_dict))

    def test_or_not_matching(self):
        filter_dict = {"$or": [{"genre": "comedy"}, {"year": 2019}]}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertFalse(filter_item(filter_dict, metadata_dict))

    def test_and_or_not_matching(self):
        filter_dict = {"$and": [{"genre": "action"}, {"$or": [{"year": 2019}, {"length_hrs": 2}]}]}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertFalse(filter_item(filter_dict, metadata_dict))

    def test_list_metadata_not_matching(self):
        filter_dict = {"genre": "drama"}
        metadata_dict = {"genre": ["comedy", "documentary"]}
        self.assertFalse(filter_item(filter_dict, metadata_dict))

    def test_in_list_metadata_not_matching(self):
        filter_dict = {"genre": {"$in": ["drama", "action"]}}
        metadata_dict = {"genre": ["comedy", "documentary"]}
        self.assertFalse(filter_item(filter_dict, metadata_dict))

    def test_and_list_metadata_not_matching(self):
        filter_dict = {"$and": [{"genre": "comedy"}, {"genre": "drama"}]}
        metadata_dict = {"genre": ["comedy", "documentary"]}
        self.assertFalse(filter_item(filter_dict, metadata_dict))

    def test_invalid_query(self):
        filter_dict = {"genre": {"$unsupported_operator": "comedy"}}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        with self.assertRaises(ValueError):
            filter_item(filter_dict, metadata_dict)
            
    def test_empty_filter(self):
        filter_dict = {}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_empty_metadata(self):
        filter_dict = {"genre": "action"}
        metadata_dict = {}
        self.assertFalse(filter_item(filter_dict, metadata_dict))
        
    def test_empty_filter_and_metadata(self):
        filter_dict = {}
        metadata_dict = {}
        self.assertTrue(filter_item(filter_dict, metadata_dict))

    def test_nested_and_or(self):
        filter_dict = {
            "$and": [
                {"$or": [{"genre": "action"}, {"genre": "comedy"}]},
                {"$or": [{"year": 2020}, {"length_hrs": 1.5}]}
            ]
        }
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertTrue(filter_item(filter_dict, metadata_dict))
        
    def test_key_not_present(self):
        filter_dict = {"non_existent_key": "value"}
        metadata_dict = {"genre": "action", "year": 2020, "length_hrs": 1.5}
        self.assertFalse(filter_item(filter_dict, metadata_dict))
   
if __name__ == '__main__':
    unittest.main()

