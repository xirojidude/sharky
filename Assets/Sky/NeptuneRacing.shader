
Shader "Skybox/NeptuneRacing"
{
    Properties
    {
        iChannel0 ("tex2D", 2D) = "white" {}
        iChannel1 ("tex2D", 2D) = "white" {}
        iChannel2 ("tex2D", 2D) = "white" {}
        iChannel3 ("tex2D", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D iChannel0; 
            uniform sampler2D iChannel1; 
            uniform sampler2D iChannel2; 
            uniform sampler2D iChannel3; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

                return o;
            }

// Neptune Racing. December 2014
// https://www.shadertoy.com/view/XtX3Rr


// Uses sphere tracing to accumulate direction normals across the landscape.
// Materials are calculated after the tracing loop from alphas and distances
// stored in a stack.
// Change ScoopRadius for depth of field.

#define PI 3.141596
float3 sunLight  = normalize( float3(  0.35, 0.2,  0.3 ) );
float3 moon  = float3(  45000., -30000.0,  -30000. );
const float3 sunColour = float3(.4, .6, 1.);
#define FOG_COLOUR float3(0.07, 0.05, 0.05)
float4 aStack[2];
float4 dStack[2];
float2 fcoord;


//--------------------------------------------------------------------------
float Hash(float2 p)
{
    return frac(sin(dot(p, float2(12.9898, 78.233))) * 33758.5453)-.5;
}

//--------------------------------------------------------------------------
float Noise(in float3 x)
{
    float3 p = floor(x);
    float3 f = frac(x);
    f = f*f*(3.0-2.0*f);
    float2 uv = (p.xy+float2(37.0,17.0)*p.z) + f.xy;
    float2 rg = tex2Dlod( iChannel2, float4((uv+ 0.5)/256.0, 0.0,0) ).yx;  //tex2D( iChannel2, (uv+ 0.5)/256.0 ).yx;  //tex2DLod( iChannel2, (uv+ 0.5)/256.0, 0.0 ).yx;

    return lerp( rg.x, rg.y, f.z );
}

//-----------------------------------------------------------------------------------
const float3x3 m = float3x3( 0.00,  0.80,  0.60,
                    -0.80,  0.46, -0.48,
                    -0.60, -0.38,  0.64 ) * 2.43;
float Turbulence( float3 p )
{
    float f;
    f  = 0.5000*Noise( p ); p = mul(m,p);
    f += 0.2500*Noise( p ); p = mul(m,p);
    f += 0.1250*Noise( p ); p = mul(m,p);
    f += 0.0625*Noise( p ); p = mul(m,p);
    f += 0.0312*Noise( p ); 
    return f;
}

//--------------------------------------------------------------------------
float SphereIntersect( in float3 ro, in float3 rd, in float4 sph )
{
    float3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    if( h<0.0 ) return -1.0;
    return -b - sqrt( h );
}

//-----------------------------------------------------------------------------------
float Terrain( in float2 q, float bias )
{
    float tx1 = smoothstep( 0.,.4, tex2Dlod( iChannel0, float4(0.000015*q,0, bias) ).y);   //tex2D( iChannel0, 0.000015*q ).y);      // tex2DLod( iChannel0, 0.000015*q, bias ).y);
    tx1   = lerp(tx1, tex2Dlod( iChannel1, float4(0.00003*q,0, bias) ).x, tx1);  //tex2D( iChannel1, 0.00003*q ).x, tx1);       //tex2DLod( iChannel1, 0.00003*q, bias ).x, tx1);
    return tx1*355.0;
}


//--------------------------------------------------------------------------
float Map( in float3 p )
{
    float h = Terrain( p.xz, -100.0 );
    float  turb =Turbulence( p * float3(1.0, 1., 1.0)*.05 ) * 25.3;
    return p.y-h+turb;
}
//--------------------------------------------------------------------------
// Grab all sky information for a given ray from camera
float3 GetSky(in float3 rd)
{
    float sunAmount = max( dot( rd, sunLight), 0.0 );
    float v = pow(1.0-max(rd.y,0.0),4.);
    float3  sky = lerp(float3(.0,0.01,.04), float3(.1, .04, .07), v);
    //sky *= smoothstep(-0.3, .0, rd.y);
    sky = sky + sunColour * sunAmount * sunAmount * .15;
    sky = sky + sunColour * min(pow(sunAmount, 1800.0), .3);
    return clamp(sky, 0.0, 1.0);
}
//--------------------------------------------------------------------------
 float3 GetClouds(float3 p,float3 dir)
 {
    float n = (1900.0-p.y)/dir.y;
    float2 p2 = p.xz + dir.xz * n;
    float3 clo = tex2Dlod(iChannel3, float4(p2*.00001+.2,0,0.) ).zyz * .04;           // tex2D(iChannel3, p2*.00001+.2).zyz * .04;        //tex2DLod(iChannel3, p2*.00001+.2,0.).zyz * .04;
    n = (1000.0-p.y)/dir.y;
    p2 = p.xz + dir.xz * n;
    clo +=   tex2Dlod(iChannel0, float4(p2*.00001-.4,0, 0.0) ).zyz * .04;       //tex2D(iChannel0, p2*.00001-.4).zyz * .04;        //tex2DLod(iChannel0, p2*.00001-.4, 0.0).zyz * .04;
    clo = clo * pow(max(dir.y, 0.0), .8)*3.0;
     return clo;

 }

//--------------------------------------------------------------------------
float ScoopRadius(float t)
{
    if (t< 150.0) t = abs(t-150.) * 3.;
    t = t*0.006;
    return clamp(t*t, 256.0/450, 20000.0/450);  //clamp(t*t, 256.0/iResolution.y, 20000.0/iResolution.y);
}

//--------------------------------------------------------------------------
// Calculate sun light...
float3 DoLighting(in float3 mat, in float3 normal, in float3 eyeDir, in float d,in float3 sky)
{
    float h = dot(sunLight,normal);
    mat = mat * sunColour*(max(h, 0.0));
    mat += float3(0.01, .01,.02) * max(normal.y, 0.0);
    normal = reflect(eyeDir, normal);
    mat += pow(max(dot(sunLight, normal), 0.0), 50.0)  * sunColour * .5;
    mat = lerp(sky,mat, min(exp(-d*d*.000002), 1.0));
    return mat;
}

//--------------------------------------------------------------------------
float3 GetNormal(float3 p, float sphereR)
{
    float2 eps = float2(sphereR*.5, 0.0);
    return normalize( float3(
           Map(p+eps.xyy) - Map(p-eps.xyy),
           Map(p+eps.yxy) - Map(p-eps.yxy),
           Map(p+eps.yyx) - Map(p-eps.yyx) ) );
}

//--------------------------------------------------------------------------
float Scene(in float3 rO, in float3 rD)
{
    //float t = 0.0;
    float t = 8.0 * Hash(fcoord);
    float  alphaAcc = 0.0;
    float3 p = float3(0.0,0,0);
    int hits = 0;
    [loop]
    for( int j=0; j < 95; j++ )
    {
        if (hits == 8  || t > 1250.0) break;
        p = rO + t*rD;
        float sphereR = ScoopRadius(t);
        float h =Map(p);
        if(h < sphereR)
        {
            // Accumulate the alphas...
            float alpha = (1.0 - alphaAcc) * min(((sphereR-h) / sphereR), 1.0);
            // If high enough to contribute nicely...
            if (alpha > (1./8.0))
            {
                // If a peice of the lanscape is scooped as a suitable alpha,
                // then it's put on the stacks...
            // put it on the 2 stacks, alpha and distance...
                aStack[1].yzw = aStack[1].xyz; aStack[1].x = aStack[0].w;
                aStack[0].yzw = aStack[0].xyz; aStack[0].x = alpha;
                dStack[1].yzw = dStack[1].xyz; dStack[1].x = dStack[0].w;
                dStack[0].yzw = dStack[0].xyz; dStack[0].x = t;
                alphaAcc += alpha;  
                hits++;
            }
            
        }
        t +=  h * .5 + t * 0.004;
       
    }
    
    return clamp(alphaAcc, 0.0, 1.0);
}





//--------------------------------------------------------------------------
float3 PostEffects(float3 rgb, float2 xy)
{
    // Gamma first...
    rgb = pow(rgb, float3(0.45,0.45,0.45));

    // Then...
    #define CONTRAST 1.1
    #define SATURATION 1.4
    #define BRIGHTNESS 1.2
    float dt = dot(float3(.2125, .7154, .0721), rgb*BRIGHTNESS);
    rgb = lerp(float3(.5,.5,.5), lerp(float3(dt,dt,dt), rgb*BRIGHTNESS, SATURATION), CONTRAST);

    // Vignette...
    rgb *= .5+0.5*pow(180.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.3 ); 

    return clamp(rgb, 0.0, 1.0);
}

//--------------------------------------------------------------------------
float3 TexCube( sampler2D sam, in float3 p, in float3 n )
{
    float3 x = tex2D( sam, p.yz ).xyz;
    float3 y = tex2D( sam, p.zx ).xyz;
    float3 z = tex2D( sam, p.xy ).xyz;
    return (x*abs(n.x) + y*abs(n.y) + z*abs(n.z))/(abs(n.x)+abs(n.y)+abs(n.z));
}
//--------------------------------------------------------------------------
float3 Albedo(float3 pos, float3 nor)
{
    float3 col = TexCube(iChannel1, pos*.01, nor).xzy + TexCube(iChannel3, pos*.02, nor);
    return col * .5;
}

//--------------------------------------------------------------------------
 float cross2(float2 A, float2 B)
 {
    return A.x*B.y-A.y*B.x;
 }

//--------------------------------------------------------------------------
 float GetAngle(float2 A, float2 B)
 {
    return atan2( dot(A,B),cross2(A,B));
}

float mod(float a, float b) {return a%b;}

//--------------------------------------------------------------------------
float3 CameraPath( float t )
{
    float s = smoothstep(.0, 3.0, t);
    float3 pos = float3( t*30.0*s +120.0, 1.0, t* 220.* s -80.0);
    
    float a = t/4.0;
    pos.xz += float2(1350.0 * cos(a), 350.0*sin(a));
    pos.xz += float2(1400.0 * sin(-a*1.8), 400.0*cos(-a*4.43));

    return pos;
} 


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(iChannel0, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin


    fcoord = fragCoord;
//    float m = 0.0;//(iMouse.x/iResolution.x)*10.0;
//    float gTime = ((_Time.y+135.0));  //*.25+m);
//    float2 xy = fragCoord.xy / iResolution.xy;
//    float2 uv = (-1.0 + 2.0 * xy) * float2(iResolution.x/iResolution.y,1.0);
    
//    float hTime = mod(gTime+1.95, 2.0);
    
    float3 cameraPos  = ro;  //CameraPath(gTime + 0.0);
//    float3 camTarget  = CameraPath(gTime + .25);
//    float3 far        = CameraPath(gTime + .4);
    

//    float2 v1 = normalize(far.xz-cameraPos.xz);
//    float2 v2 = normalize(camTarget.xz-cameraPos.xz);
//    float roll = clamp(GetAngle(v1 , v2), -.8, .8);
    
    
//    float t = Terrain(cameraPos.xz, .0)+13.0;
//    float t2 = Terrain(camTarget.xz, .0)+13.0;
//    cameraPos.y = camTarget.y= t;
    
    //roll = roll;
//    float3 cw = normalize(camTarget-cameraPos);
//    float3 cp = float3(sin(roll), cos(roll),0.0);
//    float3 cu = normalize(cross(cw,cp));
//    float3 cv = cross(cu,cw);
    float3 dir = rd; //normalize(uv.x*cu + uv.y*cv + 1.1*cw);

    float3 col = float3(0.0,0,0);
    
    for (int i = 0; i <2; i++)
    {
        dStack[i] = float4(-20.0,-20,-20,-20);
        aStack[i] = float4(0.0,0,0,0);
    }
    float alpha = Scene(cameraPos, dir);
    
     float3 sky = GetSky(dir);
    // Render both stacks...
    for (int s = 0; s < 2; s++)
    {
        for (int i = 0; i < 4; i++)
        {
            float d = dStack[s][i];
            if (d < .0) continue;
            float sphereR = ScoopRadius(d);
            float3 pos = cameraPos + dir * d;
            float occ = max(1.2-Turbulence(pos * float3(1.0, 1., 1.0)*.05 )*1.2, 0.0);
            float3 normal = GetNormal(pos, sphereR);
            float3 c = Albedo(pos, normal);
            col += DoLighting(c, normal, dir, d, sky)* aStack[s][i]*occ;
        }
    }
    
   col += sky *  (1.0-alpha);
    
    if (alpha < .8)
    {
        // Do a quick moon...
        float t = SphereIntersect(cameraPos, dir, float4(moon, 14000.0));
        if (t> 0.0)
        {
            float3 moo = cameraPos + t * dir;
            float3 nor = normalize(moo-moon);
            moo = TexCube(iChannel3, moo*.00001, nor)* max(dot(sunLight, nor), 0.0);
            
            sky = lerp(sky, moo, .2);
        }
        else
        {
            float stars = pow(tex2D(  iChannel2,  float2(atan2( dir.z,dir.x), dir.y*2.0)).x, 48.0)*.35;        //, -100.0
            stars *= pow(max(dir.y, 0.0), .8)*2.0;
            sky += stars;
        }
        sky += GetClouds(cameraPos, dir);
        col = lerp(sky ,col, alpha);
    }

//    col = PostEffects(col, xy) * smoothstep(.0, 2.0, _Time.y);    
    
    fragColor=float4(col,1.0);




                return fragColor;
            }

            ENDCG
        }
    }
}

