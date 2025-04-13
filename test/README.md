# Automated Tests

## Overview
This directory contains test cases and resources for validating PCD system functionality. Tests cover both offline processing of sample point clouds and unit-level verification of core algorithms.

## Test Types

### 1. Unit Tests
- Location: `src/UnitTests.cpp`
- Scope:
  - Individual component validation
  - Geometric algorithm verification
  - Edge case handling
- Dependencies: Google Test Framework

### 2. Offline Processing Tests
- Test PCD Files:
  - `Full Body *.pcd` - Ideal case (fully visible)
  - `PartiallyObstructed*.pcd` - Partial visibility
  - `PartiallyCoveredByAnotherHuman.pcd` - Multi-person scenario
  - `Out of Screen.pcd` - Outside Camera field of view scenario
