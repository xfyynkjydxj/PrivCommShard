import community as community_louvain
import networkx as nx


class CommunityDetector:

    def detect_communities(self, G):

        partition = community_louvain.best_partition(
            G,
            weight='weight',
            resolution=1.0
        )

        modularity = community_louvain.modularity(
            partition,
            G,
            weight='weight'
        )

        return partition, modularity
