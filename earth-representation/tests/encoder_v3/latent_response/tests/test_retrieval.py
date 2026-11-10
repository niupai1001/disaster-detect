import unittest
import torch
from encoder_v3.latent_response.retrieval import ExactKDIndex
from encoder_v3.latent_response.decoder import exact_knn
from scripts.enmap_download import forms


class RetrievalTests(unittest.TestCase):
    def test_boundary_ties_beyond_query_width_and_empty_inputs(self):
        centers=torch.cat((torch.ones(90,2),torch.zeros(70,2))).double()
        points=torch.tensor([[0.,0.],[.5,.5],[1.,1.]])
        actual,metrics=ExactKDIndex(centers).query(points)
        expected,_=exact_knn(points,centers)
        self.assertTrue(torch.equal(actual,expected))
        self.assertIn('library_internal_traversal_unavailable',metrics['distance_evaluation_scope'])
        self.assertEqual(ExactKDIndex(centers).query(torch.empty(0,2))[0].shape,(0,8))
        self.assertEqual(ExactKDIndex(torch.empty(0,2)).query(points)[0].shape,(3,0))

    def test_exact_matches_exhaustive_with_duplicates_and_ties(self):
        torch.manual_seed(5)
        for count in (1,8,32,333):
            centers=torch.randint(-8,8,(count,2)).double()
            points=torch.randn(37,2).double();points[:3]=0
            index=ExactKDIndex(centers,leaf_size=8)
            wanted,_=exact_knn(points,centers)
            found,metrics=index.query(points)
            self.assertTrue(torch.equal(found,wanted))
            self.assertLessEqual(metrics['distance_evaluations'],count*len(points))
        centers=torch.stack(torch.meshgrid(torch.arange(100),torch.arange(100),indexing='ij'),-1).reshape(-1,2).double()
        points=torch.tensor([[10.1,10.2],[90.1,90.2]])
        found,metrics=ExactKDIndex(centers).query(points)
        expected,_=exact_knn(points,centers)
        self.assertTrue(torch.equal(found,expected));self.assertLess(metrics['distance_evaluations'],1000)

    def test_login_parser_handles_attribute_order(self):
        result=forms('<form method="post" action="/login?a=1&amp;b=2" id="login"><input value="token" type="hidden" name="execution"><input name="username"><input name="password" value=""></form>')
        self.assertEqual(result[0]['fields']['execution'],'token')
        self.assertEqual(result[0]['action'],'/login?a=1&b=2')


    def test_large_tree_seed_bound_keeps_exact_outside_and_tie_results(self):
        torch.manual_seed(8)
        centers=torch.randint(-50,50,(4096,2)).double()
        points=torch.cat((torch.randn(23,2)*30,torch.tensor([[1000.,1000.],[-1000.,0.],[0.,0.]]))).double()
        actual,metrics=ExactKDIndex(centers).query(points)
        expected,_=exact_knn(points,centers)
        self.assertTrue(torch.equal(actual,expected))
        self.assertLess(metrics['distance_evaluations'],len(points)*len(centers))
