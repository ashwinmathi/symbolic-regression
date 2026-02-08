{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

{- |
Module      : Symbolic.Regression
Description : Symbolic regression for DataFrames using genetic programming with e-graph optimization

This module provides symbolic regression capabilities for DataFrame workflows.
Given a target column and a dataset, it evolves mathematical expressions that
predict the target variable, returning a Pareto front of expressions trading
off complexity and accuracy.

= Quick Start

@
import qualified DataFrame as D
import DataFrame.Functions ((.=))
import Symbolic.Regression

-- Load your data
df <- D.readParquet "./data/mtcars.parquet"

-- Run symbolic regression to predict 'mpg'
exprs <- fit defaultRegressionConfig mpg df

-- Use the best expression
D.derive "prediction" (last exprs) df
@

= Important Notes

All columns used in regression must be converted to 'Double' first.
Symbolic regression will by default only use the double columns.

= How It Works

1. __Genetic Programming__: Evolves a population of expression trees through
   selection, crossover, and mutation
2. __E-graph Optimization__: Uses equality saturation to discover equivalent
   expressions and simplify
3. __Parameter Optimization__: Fits numerical constants using nonlinear optimization
4. __Pareto Selection__: Returns expressions across the complexity-accuracy frontier
-}
module Symbolic.Regression (
    -- * Main API
    fit,

    -- * Configuration
    RegressionConfig (..),
    ValidationConfig (..),
    defaultRegressionConfig,
) where

import Control.Exception (throw)
import Control.Monad.State.Strict
import Data.Massiv.Array as MA hiding (forM, forM_)
import qualified Data.Text as T
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VU
import qualified DataFrame as D
import qualified DataFrame.Functions as F
import DataFrame.Internal.Expression
import System.Random

import Algorithm.EqSat.Build
import Algorithm.EqSat.DB
import Algorithm.EqSat.Egraph
import Algorithm.EqSat.Info
import Algorithm.EqSat.Queries
import Algorithm.EqSat.Simplify hiding (myCost)
import Algorithm.SRTree.Likelihoods
import Algorithm.SRTree.ModelSelection (fractionalBayesFactor)
import Control.Lens (over)
import Control.Monad (
    filterM,
    forM,
    forM_,
    replicateM,
    unless,
    when,
    (>=>),
 )
import Data.Binary (decode, encode)
import qualified Data.ByteString.Lazy as BS
import Data.Function (on)
import Data.Functor
import qualified Data.HashSet as Set
import qualified Data.IntMap.Strict as IM
import Data.List (
    intercalate,
    maximumBy,
    nub,
    zip4,
 )
import Data.List.Split (splitOn)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromJust, fromMaybe)
import Data.SRTree
import Data.SRTree.Datasets
import qualified Data.SRTree.Internal as SI
import Data.SRTree.Print
import Data.SRTree.Random
import Data.Type.Equality (TestEquality (testEquality), type (:~:) (Refl))
import Type.Reflection (typeRep)

import Algorithm.EqSat (runEqSat)
import Algorithm.EqSat.SearchSR
import Data.Time.Clock.POSIX
import Text.ParseSR

{- | Configuration for the symbolic regression algorithm.

Use 'defaultRegressionConfig' as a starting point and modify fields as needed:

@
myConfig :: RegressionConfig
myConfig = defaultRegressionConfig
    { generations = 200
    , maxExpressionSize = 7
    , populationSize = 200
    }
@
-}
data RegressionConfig = RegressionConfig
    { generations :: Int
    -- ^ Number of evolutionary generations to run (default: 100)
    , maxExpressionSize :: Int
    -- ^ Maximum tree depth\/complexity for generated expressions (default: 5)
    , validationConfig :: Maybe ValidationConfig
    -- ^ The configuration for cross validation.
    , showTrace :: Bool
    -- ^ Whether to print progress during evolution (default: 'True')
    , lossFunction :: Distribution
    -- ^ Loss function to optimize: 'MSE', 'Gaussian', 'Poisson', etc. (default: 'MSE')
    , numOptimisationIterations :: Int
    -- ^ Number of iterations for parameter optimization (default: 30)
    , numParameterRetries :: Int
    -- ^ Number of retries for parameter fitting (default: 2)
    , populationSize :: Int
    -- ^ Size of the expression population (default: 100)
    , tournamentSize :: Int
    -- ^ Number of individuals in tournament selection (default: 3)
    , crossoverProbability :: Double
    -- ^ Probability of crossover between expressions (default: 0.95)
    , mutationProbability :: Double
    -- ^ Probability of mutation (default: 0.3)
    , unaryFunctions :: [D.Expr Double -> D.Expr Double]
    -- ^ Unary operations to include in the search space (default: @[]@)
    , binaryFunctions :: [D.Expr Double -> D.Expr Double -> D.Expr Double]
    {- ^ Binary operations to include in the search space
    (default: @[(+), (-), (*), (\/)]@)
    -}
    , numParams :: Int
    -- ^ Number of parameters to use. Set to @-1@ for automatic detection (default: -1)
    , generational :: Bool
    -- ^ Whether to use generational replacement strategy (default: 'False')
    , simplifyExpressions :: Bool
    -- ^ Whether to simplify output expressions using e-graph optimization (default: 'True')
    , maxTime :: Int
    -- ^ Time limit in seconds. Set to @-1@ for no limit (default: -1)
    , dumpTo :: String
    -- ^ File path to save e-graph state for later resumption (default: @\"\"@)
    , loadFrom :: String
    -- ^ File path to load e-graph state from a previous run (default: @\"\"@)
    , seed :: Maybe Int
    -- ^ Random seed for deterministic results. Set to 'Nothing' for non-deterministic (default: 'Nothing')
    }

data ValidationConfig = ValidationConfig
    { validationPercent :: Double
    , validationSeed :: Int
    }

{- | Default configuration for symbolic regression.

Provides sensible defaults for most use cases:

* 100 generations with population size 100
* Maximum expression size of 5
* 3-fold cross-validation
* MSE loss function
* Basic arithmetic operations: @+@, @-@, @*@, @\/@

Modify specific fields to customize the search behavior.
-}
defaultRegressionConfig :: RegressionConfig
defaultRegressionConfig =
    RegressionConfig
        { generations = 100
        , maxExpressionSize = 5
        , validationConfig = Nothing
        , showTrace = True
        , lossFunction = MSE
        , numOptimisationIterations = 30
        , numParameterRetries = 2
        , populationSize = 100
        , tournamentSize = 3
        , crossoverProbability = 0.95
        , mutationProbability = 0.3
        , unaryFunctions = [(`F.pow` 2), (`F.pow` 3), log, (1 /)]
        , binaryFunctions = [(+), (-), (*), (/)]
        , numParams = -1
        , generational = False
        , simplifyExpressions = True
        , maxTime = -1
        , dumpTo = ""
        , loadFrom = ""
        , seed = Nothing
        }

{- | Run symbolic regression to discover mathematical expressions that fit the data.

Returns a list of expressions representing the Pareto front, ordered by
complexity (simplest first). Each expression:

* Is a valid @'D.Expr' 'Double'@ that can be used with DataFrame operations
* Represents a different trade-off between simplicity and accuracy
* Has optimized numerical constants

= Example

@
exprs <- fit defaultRegressionConfig targetColumn df

-- View discovered expressions
map D.prettyPrint exprs
-- [\"qsec\", \"57.33 \/ wt\", \"10.75 + (1557.67 \/ disp)\"]

-- Use expressions in DataFrame operations
D.derive \"prediction\" (last exprs) df
@

= Important

All columns must be converted to 'Double' before running regression.
The algorithm will only use double-typed columns as features.
-}
fit ::
    -- | Configuration controlling the search algorithm
    RegressionConfig ->
    -- | Target column expression to predict
    D.Expr Double ->
    -- | Input DataFrame containing features and target
    D.DataFrame ->
    -- | Pareto front of expressions, ordered simplest to most complex
    IO [D.Expr Double]
fit cfg targetColumn df = do
    g <- case seed cfg of
        Nothing -> getStdGen
        Just s -> pure $ mkStdGen s
    let
        (train, validation) = case validationConfig cfg of
            Nothing -> (df, df)
            Just vcfg ->
                D.randomSplit (mkStdGen (validationSeed vcfg)) (1 - validationPercent vcfg) df
        cols =
            D.columnNames
                ( D.exclude
                    [F.name targetColumn]
                    (D.selectBy [D.byProperty (D.hasElemType @Double)] train)
                )
        toFeatureMatrix d =
            either throw id $
                D.toDoubleMatrix $
                    D.exclude
                        [F.name targetColumn]
                        (D.selectBy [D.byProperty (D.hasElemType @Double)] d)
        toFeatures d =
            fromLists' Seq (V.toList (V.map VU.toList (toFeatureMatrix d))) ::
                Array S Ix2 Double
        toTarget d = fromLists' Seq (D.columnAsList targetColumn d) :: Array S Ix1 Double
        nonterminals =
            intercalate
                ","
                ( Prelude.map
                    (toNonTerminal . (\f -> f (F.col "fake1") (F.col "fake2")))
                    (binaryFunctions cfg)
                    ++ Prelude.map (toNonTerminal . (\f -> f (F.col "fake1"))) (unaryFunctions cfg)
                )
        varnames =
            intercalate
                ","
                ( Prelude.map
                    T.unpack
                    cols
                )
        alg =
            evalStateT
                ( egraphGP
                    cfg
                    nonterminals
                    varnames
                    [
                        ( (toFeatures train, toTarget train, Nothing)
                        , (toFeatures validation, toTarget validation, Nothing)
                        )
                    ]
                    [(toFeatures df, toTarget df, Nothing)]
                )
                emptyGraph
    fmap (Prelude.map (toExpr cols)) (evalStateT alg g)

toExpr :: [T.Text] -> Fix SRTree -> Expr Double
toExpr _ (Fix (Const value)) = Lit value
toExpr cols (Fix (Var ix)) = Col (cols !! ix)
toExpr cols (Fix (Uni f value)) = case f of
    SI.Square -> F.pow (toExpr cols value) 2
    SI.Cube -> F.pow (toExpr cols value) 3
    SI.Log -> log (toExpr cols value)
    SI.Recip -> F.lit 1 / toExpr cols value
    treeOp -> error ("UNIMPLEMENTED OPERATION: " ++ show treeOp)
toExpr cols (Fix (Bin op left right)) = case op of
    SI.Add -> toExpr cols left + toExpr cols right
    SI.Sub -> toExpr cols left - toExpr cols right
    SI.Mul -> toExpr cols left * toExpr cols right
    SI.Div -> toExpr cols left / toExpr cols right
    treeOp -> error ("UNIMPLEMENTED OPERATION: " ++ show treeOp)
toExpr _ _ = error "UNIMPLEMENTED"

toNonTerminal :: D.Expr Double -> String
toNonTerminal (BinaryOp "add" _ _ _) = "add"
toNonTerminal (BinaryOp "sub" _ _ _) = "sub"
toNonTerminal (BinaryOp "mult" _ _ _) = "mul"
toNonTerminal (BinaryOp "divide" _ (Lit n :: Expr b) _) = case testEquality (typeRep @b) (typeRep @Double) of
    Nothing -> error "[Internal Error] - Reciprocal of non-double"
    Just Refl -> case n of
        1 -> "recip"
        _ -> error "Unknown reciprocal"
toNonTerminal (BinaryOp "divide" _ _ _) = "div"
toNonTerminal (BinaryOp "pow" _ _ (e :: Expr b)) = case testEquality (typeRep @b) (typeRep @Int) of
    Nothing -> error "Impossible: Raised to non-int power"
    Just Refl -> case e of
        (Lit 2) -> "square"
        (Lit 3) -> "cube"
        _ -> error "Unknown power"
toNonTerminal (UnaryOp "log" _ _) = "log"
toNonTerminal e = error ("Unsupported operation: " ++ show e)

egraphGP ::
    RegressionConfig ->
    String -> -- nonterminals
    String -> -- varnames
    [(DataSet, DataSet)] ->
    [DataSet] ->
    StateT EGraph (StateT StdGen IO) [Fix SRTree]
egraphGP cfg nonterminals varnames dataTrainVals dataTests = do
    -- Phase 1: Initialize e-graph with terminal symbols
    initializeEGraph

    -- Phase 2: Create random starting population
    t0 <- io getPOSIXTime
    pop <- createInitialPopulation

    -- Phase 3: Initial trace output (if enabled)
    output <-
        if showTrace cfg
            then forM (Prelude.zip [0 ..] pop) $ uncurry printExpr'
            else pure []

    -- Phase 4: Run evolution loop (tournament → crossover → mutation → rewrites)
    let mTime = if maxTime cfg < 0 then Nothing else Just (fromIntegral $ maxTime cfg - 5)
    (_, _) <- runEvolutionLoop t0 mTime pop output

    -- Phase 5: Extract Pareto front and save results
    finalizeAndExtractPareto
  where
    -- ============================================================================
    -- CONFIGURATION & SETUP
    -- ============================================================================
    
    maxMem = 2000000
    
    -- Fitness function with parameter optimization
    fitFun =
        fitnessMV
            shouldReparam
            (numParameterRetries cfg)
            (numOptimisationIterations cfg)
            (lossFunction cfg)
            dataTrainVals
    
    -- Parse function operators from config
    nonTerms = parseNonTerms nonterminals
    
    -- Extract number of features from dataset
    (Sz2 _ nFeats) = case dataTrainVals of
        [] -> Sz2 0 0
        (h : _) -> MA.size (getX . fst $ h)
    
    -- Parameter configuration
    params =
        if numParams cfg == -1
            then [param 0]  -- Auto-detect parameters
            else Prelude.map param [0 .. numParams cfg - 1]  -- Fixed number
    shouldReparam = numParams cfg == -1
    relabel = if shouldReparam then relabelParams else relabelParamsOrder
    
    -- Terminal symbols (variables or parameters depending on loss function)
    terms =
        if lossFunction cfg == ROXY
            then var 0 : params
            else [var ix | ix <- [0 .. nFeats - 1]]
    
    -- Filter operators by arity
    uniNonTerms = [t | t <- nonTerms, isUni t]
    binNonTerms = [t | t <- nonTerms, isBin t]

    isUni (Uni _ _) = True
    isUni _ = False

    isBin (Bin{}) = True
    isBin _ = False

    -- ============================================================================
    -- MAIN PIPELINE FUNCTIONS
    -- ============================================================================
    -- ============================================================================
    -- MAIN PIPELINE FUNCTIONS
    -- ============================================================================

    -- | Initialize the e-graph: load from file if specified, insert base terms, and evaluate
    initializeEGraph :: StateT EGraph (StateT StdGen IO) ()
    initializeEGraph = do
        -- Load saved e-graph state if specified
        unless (null (loadFrom cfg)) $
            io (BS.readFile (loadFrom cfg)) >>= \eg -> put (decode eg)
        -- Insert base terminal symbols and evaluate their fitness
        _ <- insertTerms
        evaluateUnevaluated fitFun

    -- | Create initial random population
    createInitialPopulation :: StateT EGraph (StateT StdGen IO) [EClassId]
    createInitialPopulation = do
        pop <- replicateM (populationSize cfg) $ do
            -- Generate random expression tree
            ec <- insertRndExpr (maxExpressionSize cfg) rndTerm rndNonTerm >>= canonical
            -- Evaluate fitness if not already evaluated
            _ <- updateIfNothing fitFun ec
            pure ec
        Prelude.mapM canonical pop

    -- | Run the main evolution loop for the specified number of generations
    runEvolutionLoop ::
        POSIXTime ->
        Maybe NominalDiffTime ->
        [EClassId] ->
        [[String]] ->
        StateT EGraph (StateT StdGen IO) ([EClassId], [[String]])
    runEvolutionLoop t0 mTime pop output = do
        -- Helper: iterate with time limit checking
        let iterateFor 0 _ _ xs _ = pure xs
            iterateFor n t0' maxT xs f = do
                xs' <- f n xs
                t1 <- io getPOSIXTime
                let delta = t1 - t0'
                    maxT' = subtract delta <$> maxT
                case maxT' of
                    Nothing -> iterateFor (n - 1) t1 maxT' xs' f
                    Just mt ->
                        if mt <= 0
                            then pure xs  -- Time limit exceeded
                            else iterateFor (n - 1) t1 maxT' xs' f

        (finalPop, finalOutput, _) <-
            iterateFor (generations cfg) t0 mTime (pop, output, populationSize cfg) $
                \_ (ps', out, curIx) -> do
                    -- Evolve new generation
                    newPop' <- replicateM (populationSize cfg) (evolve ps')

                    -- Print trace if enabled
                    out' <-
                        if showTrace cfg
                            then forM (Prelude.zip [curIx ..] newPop') $ uncurry printExpr'
                            else pure []

                    -- Memory management: clean if e-graph gets too large
                    totSz <- gets (Map.size . _eNodeToEClass)
                    let full = totSz > max maxMem (populationSize cfg)
                    when full (cleanEGraph >> cleanDB)

                    -- Selection strategy: generational or pareto-based
                    newPop <-
                        if generational cfg
                            then Prelude.mapM canonical newPop'  -- Keep all new individuals
                            else do
                                -- Keep pareto front + best remaining individuals
                                pareto <-
                                    concat <$> forM [1 .. maxExpressionSize cfg] (`getTopFitEClassWithSize` 2)
                                let remainder = populationSize cfg - length pareto
                                lft <-
                                    if full
                                        then getTopFitEClassThat remainder (const True)
                                        else pure $ Prelude.take remainder newPop'
                                Prelude.mapM canonical (pareto <> lft)
                    pure (newPop, out <> out', curIx + populationSize cfg)

        pure (finalPop, finalOutput)

    -- | Finalize e-graph and extract Pareto front
    finalizeAndExtractPareto :: StateT EGraph (StateT StdGen IO) [Fix SRTree]
    finalizeAndExtractPareto = do
        -- Save e-graph state if specified
        unless (null (dumpTo cfg)) $
            get >>= (io . BS.writeFile (dumpTo cfg) . encode)
        -- Extract and return Pareto optimal expressions
        paretoFront' fitFun (maxExpressionSize cfg)

    -- ============================================================================
    -- MEMORY MANAGEMENT
    -- ============================================================================

    -- ============================================================================
    -- MEMORY MANAGEMENT
    -- ============================================================================

    -- Clean e-graph by keeping only top performers and rebuilding
    cleanEGraph = do
        let nParetos = 10
        io . putStrLn $ "cleaning"
        -- Extract top expressions from each complexity level
        pareto <-
            forM [1 .. maxExpressionSize cfg] (`getTopFitEClassWithSize` nParetos)
                >>= Prelude.mapM canonical . concat
        -- Save their fitness info
        infos <- forM pareto (\c -> gets (fmap _info . (IM.!? c) . _eClass))
        exprs <- forM pareto getBestExpr
        -- Reset e-graph and reinsert top performers
        put emptyGraph
        newIds <- fromTrees myCost $ Prelude.map relabel exprs
        forM_ (Prelude.zip newIds (Prelude.reverse infos)) $ \(eId, info') ->
            case info' of
                Nothing -> pure ()
                Just i'' -> insertFitness eId (fromJust $ _fitness i'') (_theta i'')

    -- ============================================================================
    -- RANDOM GENERATORS
    -- ============================================================================

    -- Generate random terminal (variable or parameter)
    rndTerm = do
        coin <- toss
        if coin || numParams cfg == 0 then randomFrom terms else randomFrom params

    -- Generate random non-terminal (operator)
    rndNonTerm = randomFrom nonTerms

    -- ============================================================================
    -- GENETIC OPERATORS
    -- ============================================================================

    -- Refit expressions that changed due to rewrites
    -- ============================================================================
    -- GENETIC OPERATORS
    -- ============================================================================

    -- Refit expressions that changed due to rewrites
    refitChanged = do
        ids <-
            (gets (_refits . _eDB) >>= Prelude.mapM canonical . Set.toList)
                Data.Functor.<&> nub
        modify' $ over (eDB . refits) (const Set.empty)
        forM_ ids $ \ec -> do
            t <- getBestExpr ec
            (f, p) <- fitFun t
            insertFitness ec f p

    -- Single evolution step: tournament → crossover → mutation → rewrite
    evolve xs' = do
        xs <- Prelude.mapM canonical xs'
        parents' <- tournament xs
        offspring <- combine parents'
        -- Apply equality saturation rewrites
        if numParams cfg == 0
            then runEqSat myCost rewritesWithConstant 1 >> cleanDB >> refitChanged
            else runEqSat myCost rewritesParams 1 >> cleanDB >> refitChanged
        canonical offspring >>= updateIfNothing fitFun >> pure ()
        canonical offspring

    -- Tournament selection: pick two parents via separate tournaments
    tournament xs = do
        p1 <- applyTournament xs >>= canonical
        p2 <- applyTournament xs >>= canonical
        pure (p1, p2)

    -- Run a single tournament: pick best from random challengers
    applyTournament :: [EClassId] -> RndEGraph EClassId
    applyTournament xs = do
        challengers <-
            replicateM (tournamentSize cfg) (rnd $ randomFrom xs) >>= traverse canonical
        fits <- Prelude.map fromJust <$> Prelude.mapM getFitness challengers
        pure . snd . maximumBy (compare `on` fst) $ Prelude.zip fits challengers

    -- Combine two parents: crossover then mutate
    combine (p1, p2) = crossover p1 p2 >>= mutate >>= canonical

    -- Crossover: swap random subtree from p2 into p1
    crossover p1 p2 = do
        sz <- getSize p1
        coin <- rnd $ tossBiased (crossoverProbability cfg)
        if sz == 1 || not coin
            then rnd (randomFrom [p1, p2])  -- No crossover
            else do
                pos <- rnd $ randomRange (1, sz - 1)  -- Random position in p1
                cands <- getAllSubClasses p2  -- All subtrees from p2
                tree <- getSubtree pos 0 Nothing [] cands p1  -- Replace subtree
                fromTree myCost (relabel tree) >>= canonical

    -- Navigate to subtree position and replace with candidate
    -- Navigate to subtree position and replace with candidate
    getSubtree ::
        Int ->
        Int ->
        Maybe (EClassId -> ENode) ->
        [Maybe (EClassId -> ENode)] ->
        [EClassId] ->
        EClassId ->
        RndEGraph (Fix SRTree)
    -- Base case: found position, replace with random valid candidate
    getSubtree 0 sz (Just parent) mGrandParents cands p' = do
        p <- canonical p'
        candidates' <-
            filterM (fmap (< maxExpressionSize cfg - sz) . getSize) cands
        candidates <-
            filterM (doesNotExistGens mGrandParents . parent) candidates'
                >>= traverse canonical
        if null candidates
            then getBestExpr p
            else do
                subtree <- rnd (randomFrom candidates)
                getBestExpr subtree
    -- Recursive case: navigate down tree
    getSubtree pos sz parent mGrandParents cands p' = do
        p <- canonical p'
        root <- getBestENode p >>= canonize
        case root of
            Param ix -> pure . Fix $ Param ix
            Const x -> pure . Fix $ Const x
            Var ix -> pure . Fix $ Var ix
            Uni f t' -> do
                t <- canonical t'
                Fix . Uni f
                    <$> getSubtree (pos - 1) (sz + 1) (Just $ Uni f) (parent : mGrandParents) cands t
            Bin op l'' r'' -> do
                l <- canonical l''
                r <- canonical r''
                szLft <- getSize l
                szRgt <- getSize r
                -- Position in right subtree
                if szLft < pos
                    then do
                        l' <- getBestExpr l
                        r' <-
                            getSubtree
                                (pos - szLft - 1)
                                (sz + szLft + 1)
                                (Just $ Bin op l)
                                (parent : mGrandParents)
                                cands
                                r
                        pure . Fix $ Bin op l' r'
                    -- Position in left subtree
                    else do
                        l' <-
                            getSubtree
                                (pos - 1)
                                (sz + szRgt + 1)
                                (Just (\t -> Bin op t r))
                                (parent : mGrandParents)
                                cands
                                l
                        r' <- getBestExpr r
                        pure . Fix $ Bin op l' r'

    -- Extract all equivalence classes (subtrees) from an expression
    getAllSubClasses p' = do
        p <- canonical p'
        en <- getBestENode p
        case en of
            Bin _ l r -> do
                ls <- getAllSubClasses l
                rs <- getAllSubClasses r
                pure (p : ls <> rs)
            Uni _ t -> (p :) <$> getAllSubClasses t
            _ -> pure [p]

    -- Mutation: randomly replace a subtree
    mutate p = do
        sz <- getSize p
        coin <- rnd $ tossBiased (mutationProbability cfg)
        if coin
            then do
                pos <- rnd $ randomRange (0, sz - 1)  -- Random position
                tree <- mutAt pos (maxExpressionSize cfg) Nothing p
                fromTree myCost (relabel tree) >>= canonical
            else pure p  -- No mutation

    -- Strip children from tree node (keep only structure)
    peel :: Fix SRTree -> SRTree ()
    peel (Fix (Bin op _ _)) = Bin op () ()
    peel (Fix (Uni f _)) = Uni f ()
    peel (Fix (Param ix)) = Param ix
    peel (Fix (Var ix)) = Var ix
    peel (Fix (Const x)) = Const x

    -- Mutate at specific position in tree
    mutAt ::
        Int -> Int -> Maybe (EClassId -> ENode) -> EClassId -> RndEGraph (Fix SRTree)
    mutAt 0 sizeLeft Nothing _ = insertRndExpr sizeLeft rndTerm rndNonTerm >>= canonical >>= getBestExpr
    mutAt 0 1 _ _ = rnd $ randomFrom terms
    mutAt 0 sizeLeft (Just parent) _ = do
        ec <- insertRndExpr sizeLeft rndTerm rndNonTerm >>= canonical
        (Fix tree) <- getBestExpr ec
        root <- getBestENode ec
        exist <- canonize (parent ec) >>= doesExist
        if exist
            then do
                -- Filter valid replacement operators based on arity
                let children = childrenOf root
                candidates <- case length children of
                    0 ->
                        filterM
                            (checkToken parent . replaceChildren children)
                            (Prelude.map peel terms)
                    1 -> filterM (checkToken parent . replaceChildren children) uniNonTerms
                    2 -> filterM (checkToken parent . replaceChildren children) binNonTerms
                    _ -> pure []
                if null candidates
                    then pure $ Fix tree
                    else do
                        newToken <- rnd (randomFrom candidates)
                        pure . Fix $ replaceChildren (childrenOf tree) newToken
            else pure . Fix $ tree
    -- Recursive case: navigate to mutation position
    mutAt pos sizeLeft _ p' = do
        p <- canonical p'
        root <- getBestENode p >>= canonize
        case root of
            Param ix -> pure . Fix $ Param ix
            Const x -> pure . Fix $ Const x
            Var ix -> pure . Fix $ Var ix
            Uni f t' ->
                canonical t'
                    >>= ( fmap (Fix . Uni f)
                            . mutAt (pos - 1) (sizeLeft - 1) (Just $ Uni f)
                        )
            Bin op ln rn -> do
                l <- canonical ln
                r <- canonical rn
                szLft <- getSize l
                szRgt <- getSize r
                -- Position in right subtree
                if szLft < pos
                    then do
                        l' <- getBestExpr l
                        r' <- mutAt (pos - szLft - 1) (sizeLeft - szLft - 1) (Just $ Bin op l) r
                        pure . Fix $ Bin op l' r'
                    -- Position in left subtree
                    else do
                        l' <- mutAt (pos - 1) (sizeLeft - szRgt - 1) (Just (\t -> Bin op t r)) l
                        r' <- getBestExpr r
                        pure . Fix $ Bin op l' r'

    -- ============================================================================
    -- OUTPUT & REPORTING
    -- ============================================================================

    -- Format expression with fitness metrics for trace output
    printExpr' :: Int -> EClassId -> RndEGraph [String]
    printExpr' ix ec' = do
        ec <- canonical ec'
        thetas' <- gets (fmap (_theta . _info) . (IM.!? ec) . _eClass)
        bestExpr <-
            (if simplifyExpressions cfg then simplifyEqSatDefault else id)
                <$> getBestExpr ec

        -- Relabel parameters and check if refitting needed
        let best' =
                if shouldReparam then relabelParams bestExpr else relabelParamsOrder bestExpr
            nParams' = countParamsUniq best'
            fromSz (MA.Sz x) = x
            nThetas = fmap (Prelude.map (fromSz . MA.size)) thetas'
        (_, thetas) <-
            if maybe False (Prelude.any (/= nParams')) nThetas
                then fitFun best'
                else pure (1.0, fromMaybe [] thetas')

        -- Calculate metrics for each data split
        maxLoss <- negate . fromJust <$> getFitness ec
        forM (Data.List.zip4 [(0 :: Int) ..] dataTrainVals dataTests thetas) $ \(view, (dataTrain, dataVal), dataTest, theta') -> do
            let (x, y, mYErr) = dataTrain
                (x_val, y_val, mYErr_val) = dataVal
                (x_te, y_te, mYErr_te) = dataTest
                distribution = lossFunction cfg

                expr = paramsToConst (MA.toList theta') best'
                showNA z = if isNaN z then "" else show z
                -- Metrics: R², NLL (negative log likelihood), MDL
                r2_train = r2 x y best' theta'
                r2_val = r2 x_val y_val best' theta'
                r2_te = r2 x_te y_te best' theta'
                nll_train = nll distribution mYErr x y best' theta'
                nll_val = nll distribution mYErr_val x_val y_val best' theta'
                nll_te = nll distribution mYErr_te x_te y_te best' theta'
                mdl_train = fractionalBayesFactor distribution mYErr x y theta' best'
                mdl_val = fractionalBayesFactor distribution mYErr_val x_val y_val theta' best'
                mdl_te = fractionalBayesFactor distribution mYErr_te x_te y_te theta' best'
                vals =
                    intercalate "," $
                        Prelude.map
                            showNA
                            [ nll_train
                            , nll_val
                            , nll_te
                            , maxLoss
                            , r2_train
                            , r2_val
                            , r2_te
                            , mdl_train
                            , mdl_val
                            , mdl_te
                            ]
                thetaStr = intercalate ";" $ Prelude.map show (MA.toList theta')
                showExprFun = if null varnames then showExpr else showExprWithVars (splitOn "," varnames)
                showLatexFun = if null varnames then showLatex else showLatexWithVars (splitOn "," varnames)
            -- CSV format: id,view,expr,python,latex,params,size,metrics...
            pure $
                show ix
                    <> ","
                    <> show view
                    <> ","
                    <> showExprFun expr
                    <> ","
                    <> "\""
                    <> showPython best'
                    <> "\","
                    <> "\"$$"
                    <> showLatexFun best'
                    <> "$$\","
                    <> thetaStr
                    <> ","
                    <> show @Int (countNodes $ convertProtectedOps expr)
                    <> ","
                    <> vals

    -- ============================================================================
    -- PARETO FRONT EXTRACTION
    -- ============================================================================

    -- Insert all terminal symbols into e-graph
    insertTerms = forM terms (fromTree myCost >=> canonical)

    -- Extract Pareto front: best expression at each complexity level
    paretoFront' _ maxSize' = go 1 (-(1.0 / 0.0))
      where
        go :: Int -> Double -> RndEGraph [Fix SRTree]
        go n f
            | n > maxSize' = pure []
            | otherwise = do
                ecList <- getBestExprWithSize n
                if not (null ecList)
                    then do
                        let (ec', mf) = case ecList of
                                [] -> (0, Nothing)
                                (e : _) -> e
                            f' = fromJust mf
                            improved = f' >= f && not (isNaN f') && not (isInfinite f')
                        ec <- canonical ec'
                        if improved
                            then do
                                -- Extract expression with fitted parameters
                                thetas' <- gets (fmap (_theta . _info) . (IM.!? ec) . _eClass)
                                bestExpr <-
                                    relabelParams . (if simplifyExpressions cfg then simplifyEqSatDefault else id)
                                        <$> getBestExpr ec
                                let t = case thetas' of
                                        Just (h : _) -> paramsToConst (MA.toList h) bestExpr
                                        _ -> Fix (Const 0) -- Not sure if this makes sense as a default.
                                ts <- go (n + 1) (max f f')
                                pure (t : ts)
                            else go (n + 1) (max f f')
                    else go (n + 1) f
