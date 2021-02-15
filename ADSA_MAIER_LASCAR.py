import random
import numpy as np
import math as maths
import pandas as pd

#####################STEP 1########################################################

class TreeNode(): 
    
    #The node is a Player object
    def __init__(self, play): 
        self.play = play
        #the key we look at is the score of the Player
        self.val=play.score
        self.left = None
        self.right = None
        self.height = 1
        
    def __str__(self):
        return("p"+self.play.id+"_score"+str(self.val))
        
    #displays the whole tree
    def display(self):
        print("")
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)
            
    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child
        if self.right is None and self.left is None:
            line = '%s' % self.val
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle
        # Only left child
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.val
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2
        # Only right child
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.val
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2
        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.val
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2
 
    
class AVL_Tree(): 
    
    def insert(self, root, key): 
        # Inserting method of the BST
        if not root: 
            return TreeNode(key) 
        elif key.score < root.val: 
            root.left = self.insert(root.left, key) 
        else: 
            root.right = self.insert(root.right, key) 
        # Update the height of the root and get the balance factor
        root.height = 1 + max(self.getHeight(root.left), 
                           self.getHeight(root.right)) 
        balance = self.getBalance(root) 
        # If the tree is unbalanced :
        #Left Left 
        if balance > 1 and key.score <= root.left.val: 
            return self.rightRotate(root) 
        # Right Right 
        if balance < -1 and key.score >= root.right.val: 
            return self.leftRotate(root) 
        # Left Right 
        if balance > 1 and key.score > root.left.val: 
            root.left = self.leftRotate(root.left) 
            return self.rightRotate(root) 
        # Right Left 
        if balance < -1 and key.score < root.right.val: 
            root.right = self.rightRotate(root.right) 
            return self.leftRotate(root) 
        return root 
    
    def leftRotate(self, z): 
        y = z.right 
        T2 = y.left 
        # Perform rotation 
        y.left = z 
        z.right = T2 
        # Update heights 
        z.height = 1 + max(self.getHeight(z.left), 
                         self.getHeight(z.right)) 
        y.height = 1 + max(self.getHeight(y.left), 
                         self.getHeight(y.right)) 
        # Return the new root 
        return y 
    
    def rightRotate(self, z): 
        y = z.left 
        T3 = y.right 
        # Perform rotation 
        y.right = z 
        z.left = T3 
        # Update heights 
        z.height = 1 + max(self.getHeight(z.left), 
                        self.getHeight(z.right)) 
        y.height = 1 + max(self.getHeight(y.left), 
                        self.getHeight(y.right)) 
        # Return the new root 
        return y 
    
    def getHeight(self, root): 
        if not root: 
            return 0
        return root.height 
    
    def getBalance(self, root): 
        if not root: 
            return 0
        return self.getHeight(root.left) - self.getHeight(root.right) 
    
    def inOrder(self, root): 
        # acending order
        if root==None:
            return []
        left_list = self.inOrder(root.right)
        right_list = self.inOrder(root.left)
        return left_list + [root.play] + right_list
    
   
    
# data structure to represesent a Player 
class Player:
    def __init__(self,id):
        self.id=id
        #pts is the number of points gained in 1 game
        self.pts=0
        self.nb_games=0
        #Score is the mean of all its games
        self.score=0
        #self.ranking=None
    def update_score(self):
        self.score=round(float((self.pts+self.score*(self.nb_games-1))/self.nb_games),1)
    def __str__(self):
        return("p"+self.id)
    def show_more(self):
        print("Player n°"+self.id
               +", score : "+str(self.score)
               +" for "+str(self.nb_games)+" games played")


class Game:
    def __init__(self,players):
        #list of 10 players
        self.players=players
        
    def assign_pts(self):
      #Assign to each player of the game a random score (number of points)
      #Update the number of games playes by the player
      for person in self.players:
          person.pts=random.randint(0,12) #Ils ont forcement des points differents?
          if (person.nb_games==None):
              person.nb_games=1
          else :
              person.nb_games+=1
          person.update_score()
          
    def __str__(self):
        res="Current Game : Player / Points "
        for player in self.players:
            res=res+"\np"+player.id+" : "+str(player.pts)+" pts"
        return res
    
 
#Database methods              
def createDB() :
    database=[]
    for i in range(1,101):
        database.append(Player(str(i)))
    print("The database has been created")
    return database 

def show(database):
    res="Database : "
    for player in database:
        res=res+"p"+str(player.id)+" "
    print(res)


#Games methods
def random_game(database,nb_game): 
    list_poss=[]
    list_players=[]
    #We have take off the players that already did a certain number of games
    for p in database:
        if p.nb_games<nb_game:
            list_poss.append(p)
    #Id of the randomly selected players
    id=random.sample(range(0,len(list_poss)),10)
    for rd_val in id:  #We take the 10 random numbers
        list_players.append(list_poss[rd_val])
    g=Game(list_players)
    #print(g)
    g.assign_pts()
    #print("points assigned")
    #print("Random game played")
    return (database) 


def create_game_ranking(database, rank_start, rank_end, willUpdate):
    print("Game for players with ranking between "+str(rank_start)+" and "+str(rank_end))
    players=[]
    for i in range(rank_start-1,rank_end):
        players=players+[database[i]]
    game=Game(players)
    #print(game)
    game.assign_pts()
    if (willUpdate==True):
        database=update_database(database)
    return(database)
    #We update the database

def update_database(database):
    tree = AVL_Tree() 
    root = None
    for players in database:
        root=tree.insert(root, players)   
    #root.display()
    #New database with players sorted depending on their score
    database=tree.inOrder(root)
    return database
    
def final_game(database):
    #The database coresponds only to the 10 best players
    print("\nFinal game")
    for i in range(5):
        database=create_game_ranking(database,1,10,True)
    tree = AVL_Tree() 
    root = None
    for players in database:
        root=tree.insert(root, players)
    #root.display()
    #New database with players sorted depending on their score
    database=tree.inOrder(root)
    print("\nTOP10 players")
    for p in database:
        p.show_more()
    print("\nPODIUM")
    print("First place :")
    database[0].show_more()
    print("Second place : ")
    database[1].show_more()
    print("Third place : ")
    database[2].show_more()
    
    return(database)

def tournament(database):
    #3 random games per player, in 3 steps
    print("Each player has to do 3 random games")
    for j in [1,2,3]:
        for i in range(10):
            database=random_game(database,j)
        print("Step "+str(j)+" : Every players did "+str(j)+" random game")
    #To sort our players we enter them in an AVL tree
    print("We create a tree and insert each player of the database")
    tree = AVL_Tree() 
    root = None
    for players in database:
        root=tree.insert(root, players)
        
    #root.display()
    #New database with players sorted depending on their score
    database=tree.inOrder(root)
    #for players in database:
     #   players.show_more()
    print("The database has been updated")
    
    
    print("\nEverybody does games depending on their ranking") 
    for j in range(9): 
        #Every time j is increasing by 1, 10 players have been ejected from the tournament
        rank_start=91-10*j
        rank_end=100-10*j
        nb_games_to_do= rank_end//10
        #print("We are going to do "+str(nb_games_to_do)+" games")
        for i in range(nb_games_to_do):
            #All players are doing 1 game depending on their ranking
            database=create_game_ranking(database, rank_start,rank_end,False) 
            #We are not updating the database yet
            #print("A game is played for players with ranking between "+str(rank_start)+" and "+str(rank_end))
            rank_start-=10
            rank_end-=10
        database=update_database(database)
        #We delete the worse 10 players
        rank_stop=len(database)-10
        database=database[:rank_stop]
        print("\nThe 10 worse players have been ejected from the tournament")
        print("Now we have "+str(len(database))+" players in the game")
    #We delete the worse 10 players
    return(database)
    
    
######################################### Step 2 ########################################################################


#The matrix of "have seen" relationships between players. 
#1 represents the fact that two players saw each other, 0 they did not see each other. 
#A player cannot see himself so for the Gii box we enter 0.

G=np.array([[0,1,0,0,1,1,0,0,0,0],
           [1,0,1,0,0,0,1,0,0,0],
           [0,1,0,1,0,0,0,1,0,0],
           [0,0,1,0,1,0,0,0,1,0],
           [1,0,0,1,0,0,0,0,0,1],
           [1,0,0,0,0,0,0,1,1,0],
           [0,1,0,0,0,0,0,0,1,1],
           [0,0,1,0,0,1,0,0,0,1],
           [0,0,0,1,0,1,1,0,0,0],
           [0,0,0,0,1,0,1,1,0,0]])
           
           
           
def potential_impostors(G1):
    for i in range(10):# a loop on the 10 players
        if i==1 or i==4 or i==5: #each in turn is supposed to be impostors.
            l=[1,2,3,4,5,6,7,8,9] #initially everyone is suspect.
            l.remove(i) #we remove the supposed impostor to leave in the list the players who are potentially in team with him.
            a=0
            for j in range(1,10):
                if G[i][j]==1:
                    l.remove(j)
                    a+=1
                    if a==2: break #1, 4 or 5 are only linked to 2 players still alive.
            print( "The potential impostors if player",i,"is impostor are :" ,l)
                
        
############################################## Step 3 #########################################################



# The matrices of impostor and crewmate graphs.
#Only the connected parts have a time, the others have an infinite time because they are not connected to each other. 
#To go from room A to this same room A the time is 0 seconds.

Crewmate=np.array([[0,2.5,2.5,2.2,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [2.5,0,3.5,2.5,3.6,maths.inf,maths.inf,4.6,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [2.5,3.5,0,2.5,maths.inf,4,5,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [2.2,2.5,2.5,0,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [maths.inf,3.6,maths.inf,maths.inf,0,maths.inf,maths.inf,3.8,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [maths.inf,maths.inf,4,maths.inf,maths.inf,0,3,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [maths.inf,maths.inf,5,maths.inf,maths.inf,3,0,4,3.5,3,maths.inf,3.2,maths.inf,maths.inf],
                   [maths.inf,4.6,maths.inf,maths.inf,3.8,maths.inf,4,0,3.7,maths.inf,maths.inf,maths.inf,2.8,maths.inf],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,3.5,3.7,0,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,3,maths.inf,maths.inf,0,maths.inf,2,maths.inf,maths.inf],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,0,4.8,2.3,3.6],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,3.2,maths.inf,maths.inf,2,4.8,0,5.2,4.2],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,2.8,maths.inf,maths.inf,2.3,5.2,0,3.7],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,3.6,4.2,3.7,0]])
                   



Impostors=np.array([[0,1.2,1.2,2.2,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [1.2,0,3.5,2.5,3.6,maths.inf,maths.inf,4.6,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [1.2,3.5,0,2.5,maths.inf,4,5,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [2.2,2.5,2.5,0,0.8,0.9,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [maths.inf,3.6,maths.inf,0.8,0,0.8,maths.inf,3.8,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [maths.inf,maths.inf,4,0.9,0.8,0,3,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf],
                   [maths.inf,maths.inf,5,maths.inf,maths.inf,3,0,4,3.5,3,maths.inf,3.2,maths.inf,maths.inf],
                   [maths.inf,4.6,maths.inf,maths.inf,3.8,maths.inf,4,0,1.6,maths.inf,4,2.9,2.8,3.5],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,3.5,1.6,0,maths.inf,3.4,2.3,3.6,2.9],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,3,maths.inf,maths.inf,0,maths.inf,2,maths.inf,maths.inf],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,4,3.4,maths.inf,0,4.8,2.3,3.6],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,3.2,2.9,2.3,2,4.8,0,5.2,1],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,2.8,3.6,maths.inf,2.3,5.2,0,0.8],
                   [maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,maths.inf,3.5,2.9,maths.inf,3.6,1,0.8,0]])
            
            
#To make the graph easier to read, we make the above matrices into dataframes.

df_Crewmate=pd.DataFrame(Crewmate,
                         ['Reactor','Upper E','Lower E','Security','Medbay','Electrical','Storage','Cafeteria','Room1','Room2','O2','Shield','Weapons','Navigations'],
                         ['Reactor','Upper E','Lower E','Security','Medbay','Electrical','Storage','Cafeteria','Room1','Room2','O2','Shield','Weapons','Navigations'])


df_Impostors=pd.DataFrame(Impostors,
                          ['Reactor','Upper E','Lower E','Security','Medbay','Electrical','Storage','Cafeteria','Room1','Room2','O2','Shield','Weapons','Navigations'],
                          ['Reactor','Upper E','Lower E','Security','Medbay','Electrical','Storage','Cafeteria','Room1','Room2','O2','Shield','Weapons','Navigations'])

#pathfinding algorithm : floyd-warshall
def floyd(n,G):
    M=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j]=G[i,j]
    for k in range(n):
        #the Dk matrix
        for i in range(n):
            for j in range(n):
                if M[i,j] > M[i,k] + M[k,j]:
                    M[i,j] = M[i,k] + M[k,j]
    
    return M    
        
pathfinding_Impostors= floyd(14,Impostors)
pathfinding_Crewmate= floyd(14,Crewmate)
df_pathfindingImpostors=pd.DataFrame(pathfinding_Impostors,['reactor','Upper E','Lower E','Security','medbay','electrical','storage','Cafeteria','Room1','Room2','O2','Shield','Weapons','Navigations'],['reactor','Upper E','Lower E','Security','medbay','electrical','storage','Cafeteria','Room1','Room2','O2','Shield','Weapons','Navigations'])
df_pathfindingCrewmate=pd.DataFrame(pathfinding_Crewmate,['reactor','Upper E','Lower E','Security','medbay','electrical','storage','Cafeteria','Room1','Room2','O2','Shield','Weapons','Navigations'],['reactor','Upper E','Lower E','Security','medbay','electrical','storage','Cafeteria','Room1','Room2','O2','Shield','Weapons','Navigations'])
Interval_time=pd.DataFrame(pathfinding_Crewmate-pathfinding_Impostors,['reactor','Upper E','Lower E','Security','medbay','electrical','storage','Cafeteria','Room1','Room2','O2','Shield','Weapons','Navigations'],['reactor','Upper E','Lower E','Security','medbay','electrical','storage','Cafeteria','Room1','Room2','O2','Shield','Weapons','Navigations'])


############################################ Step 4 ############################################

# p is the list of the visited rooms
# At the beginning paths=[]
#paths is a list of list, with all the paths possibles for the first room of the list p
def hamilton_path(Graph, p, paths):
    
    if p.count(p[-1])>1:  #If the last visited room appears more than 1 time in the visited rooms list
        return None
    if len(p)==len(Graph[0]):
        paths.append(p)
        return None
    #Look at all the neighbors from the last room visited
    voisins=proches_voisins(Graph,p[-1])
    for voisin in voisins:
        new_p=p+[voisin]
        hamilton_path(Graph,new_p,paths)

# Find all the neighbors from the current room
def proches_voisins(Graph,current_room):
    voisins=[]
    for room in range(len(Graph[0])): #for each room
        if current_room!=room and Graph[current_room][room]!=maths.inf:
            voisins.append(room)
    return voisins

#return all the paths for all the rooms
def all_paths(Graph):
    paths=[]
    for i in range(len(Graph[0])):
        hamilton_path(Graph, [i], paths)
    return paths

def weight(Graph,path):
    weight=0
    for i in range(len(path)-1):
        weight+=Graph[path[i]][path[i+1]]
    return weight
            
def min_of_paths(Graph):
    weights=[]  #list of weight of all the paths of all the rooms, 80 paths
    index=[0]   # limit indexs between which the paths are starting from the same room
    index2=[]   #index of the minimum paths for each room 
    weights2=[] #minimum weights for each starting room
    paths=all_paths(Graph)
    for path in paths:
        weights.append(weight(Graph,path))
    #the index list give the index starting from when the paths are with a different starting room, and end with the total number of paths
    for i in range(len(paths)-1):
        if paths[i][0]!=paths[i+1][0]:  #we 
            index.append(i+1)
    index.append(len(weights))  # there are as much paths as weights
    for i in range(len(index)-1):
        #we take the minimum weight from all the paths starting with the same room
        #initialisation
        min_weight=weights[index[i]]
        pos_min_path=index[i]
        #we compare the weight of all the paths starting from the same room and search the minimum
        for j in range(index[i],index[i+1]):
            if weights[j]<min_weight:
                min_weight=weights[j]
                pos_min_path=j
        weights2.append(min_weight)
        index2.append(pos_min_path)
    min_paths=[]
    for i in index2:
        min_paths.append(paths[i])   #list of all the minimum hamilton path for each distinct room
    
    return weights2, min_paths
            
weights,paths=min_of_paths(Crewmate)
name=list(df_Crewmate.columns)


#We name the colums of the matrix
for i in range(len(weights)): #for each path
    for j in range(len(paths[0])): 
        paths[i][j]=name[paths[i][j]] 
        

    
        
if __name__=="__main__":
    
    print("\n----------------------------STEP 1------------------------------\n ")
    database=createDB()
    #show(database)
    database=tournament(database)
    for p in database:
        p.show_more()
    database=final_game(database)
    
    print("\n----------------------------STEP 2--------------------------------\n ")
    potential_impostors(G)
    
    print("\n----------------------------STEP 3--------------------------------\n ")
    
    print("Matrix of time to to go from a room to another for the impostors players :")
    print(df_pathfindingImpostors)
    print("Matrix of time to to go from a room to another for the impostors players :")
    print(df_pathfindingCrewmate)
    #the advance in seconds that imposters have to move between two pieces.
    print(Interval_time)
    print("\n----------------------------STEP 4--------------------------------\n ")
    print("The shortest paths to finish all the tasks are : ")        
    for i in range(len(weights)):
        print()
        print(paths[i], "\nTime needed : " , weights[i], "seconds")    
    
    
    
    
    
    
    
    
    